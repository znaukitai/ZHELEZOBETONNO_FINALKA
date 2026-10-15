"""Mog Battle Bot: Telegram-бот для рейтинга и батлов (aiogram 3, SQLite, MediaPipe).

Весь проект в одном файле. Разделы:
  1. CONFIG        настройки (из переменных окружения) и правила эло / подбора
  2. DATABASE      SQLite (встроенный sqlite3)
  3. ELO           сколько очков стоит батл
  4. MATCHMAKING   очередь поиска соперника
  5. SCORING       оценка фото (MediaPipe Face Mesh + OpenCV)
  6. UI            клавиатуры и «окна» с картинкой
  7. HANDLERS      регистрация, меню, профиль, топы, батл
  8. ADMIN         админ-панель
  9. MAIN          запуск
"""
import asyncio
import io
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.bot import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (CallbackQuery, InlineKeyboardMarkup, InputMediaPhoto,
                           Message)
from aiogram.utils.keyboard import InlineKeyboardBuilder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mogbot")


# ==========================================================================
# 1. CONFIG
# ==========================================================================

# Secrets and settings come from environment variables (Railway -> Variables).
BOT_TOKEN = os.getenv("BOT_TOKEN", "8808641162:AAE4-Pzv6M6LXsVPO2w4kWqv2CDvqGiYELU")
ADMIN_ID = int(os.getenv("ADMIN_ID", "8206906617"))
# SQLite file. On Railway attach a Volume: its mount path is picked up automatically,
# so the data survives redeploys. Or set DB_PATH yourself (e.g. /data/mogbot.db).
DB_PATH = os.getenv("DB_PATH") or os.path.join(
    os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "."), "mogbot.db")

# Elo rules
START_ELO = 1000
ELO_STEP = 25            # win/lose when score gap <= BIG_GAP
ELO_STEP_BIG = 28        # win/lose when score gap > BIG_GAP
BIG_GAP = 1.0

# Matchmaking
MATCH_RANGE = 200        # initial +-range
MATCH_RANGE_STEP = 100   # widen by this much...
MATCH_RANGE_EVERY = 30   # ...every N seconds of waiting
MATCH_TIMEOUT = 300      # give up after N seconds

# Photos
PHOTOS_PER_BATTLE = 3

# Local time of the college (UTC+5), used for "top of the day/week"
LOCAL_TZ = timezone(timedelta(hours=5))

# Menu pictures: Telegram file_id or a DIRECT image link (https://i.ibb.co/...).
# Empty value = the screen is shown as plain text.
PHOTOS = {
    "welcome": "https://ibb.co.com/zHbGvmGQ",
    "main": "https://ibb.co.com/w8tPtr8",
    "battle": "https://ibb.co.com/YgKxMrL",
    "top": "https://ibb.co.com/B2vQ9tLz",
    "profile": "https://ibb.co.com/VYfRjgrr",
    "rules": "https://ibb.co.com/wFFS5xjM",
    "admin": "https://ibb.co.com/vvqXwjC5",
}


# ==========================================================================
# 2. DATABASE  (SQLite: один файл, отдельная база не нужна)
# ==========================================================================

# Все запросы выполняются в одном потоке event loop без await внутри,
# поэтому одного соединения достаточно, блокировок не нужно.
conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    tg_id      INTEGER PRIMARY KEY,
    student_id TEXT UNIQUE NOT NULL,
    name       TEXT NOT NULL,
    elo        INTEGER NOT NULL DEFAULT 1000 CHECK (elo >= 0),
    wins       INTEGER NOT NULL DEFAULT 0,
    losses     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);
CREATE TABLE IF NOT EXISTS elo_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id      INTEGER NOT NULL REFERENCES players(tg_id) ON DELETE CASCADE,
    delta      INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_elo_history_time ON elo_history (created_at);
"""


def _utc_text(dt: datetime) -> str:
    """datetime -> 'YYYY-MM-DD HH:MM:SS' in UTC (the format stored in created_at)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


async def db_init() -> None:
    global conn
    folder = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # SQLite LIKE is case-insensitive only for ASCII, so lowercase in Python (works for Cyrillic)
    conn.create_function("pylower", 1, lambda s: s.lower() if isinstance(s, str) else s)
    conn.executescript(SCHEMA)
    conn.commit()
    log.info("SQLite database: %s", os.path.abspath(DB_PATH))


async def db_close() -> None:
    if conn:
        conn.close()


async def db_get_player(tg_id: int):
    return conn.execute("SELECT * FROM players WHERE tg_id = ?", (tg_id,)).fetchone()


async def db_create_player(tg_id: int, student_id: str, name: str) -> bool:
    """Returns False if this student id (or telegram id) is already registered."""
    try:
        with conn:
            conn.execute(
                "INSERT INTO players (tg_id, student_id, name, elo) VALUES (?, ?, ?, ?)",
                (tg_id, student_id, name, START_ELO))
        return True
    except sqlite3.IntegrityError:
        return False


async def db_apply_result(winner_id: int, loser_id: int, delta_win: int, delta_lose: int):
    """Applies a battle result. Elo never goes below 0.
    Returns (winner_elo, loser_elo, real_loser_delta), or None if somebody was deleted."""
    with conn:  # one transaction
        w = conn.execute("SELECT elo FROM players WHERE tg_id = ?", (winner_id,)).fetchone()
        l = conn.execute("SELECT elo FROM players WHERE tg_id = ?", (loser_id,)).fetchone()
        if w is None or l is None:  # somebody was deleted mid-battle
            return None
        new_w = w["elo"] + delta_win
        new_l = max(0, l["elo"] - delta_lose)
        real_lose = new_l - l["elo"]  # <= 0
        conn.execute("UPDATE players SET elo = ?, wins = wins + 1 WHERE tg_id = ?", (new_w, winner_id))
        conn.execute("UPDATE players SET elo = ?, losses = losses + 1 WHERE tg_id = ?", (new_l, loser_id))
        conn.execute("INSERT INTO elo_history (tg_id, delta) VALUES (?, ?), (?, ?)",
                     (winner_id, delta_win, loser_id, real_lose))
    return new_w, new_l, real_lose


async def db_top_alltime(limit: int = 100):
    return conn.execute(
        "SELECT tg_id, name, elo FROM players ORDER BY elo DESC, created_at LIMIT ?",
        (limit,)).fetchall()


async def db_top_period(since: datetime, limit: int = 100):
    """Players ranked by how much elo they gained since `since` (only positive)."""
    return conn.execute(
        """
        SELECT p.tg_id AS tg_id, p.name AS name, SUM(h.delta) AS gain
        FROM elo_history h JOIN players p ON p.tg_id = h.tg_id
        WHERE h.created_at >= ?
        GROUP BY p.tg_id, p.name
        HAVING SUM(h.delta) > 0
        ORDER BY gain DESC LIMIT ?
        """,
        (_utc_text(since), limit)).fetchall()


async def db_rank_of(elo: int) -> int:
    return 1 + conn.execute("SELECT COUNT(*) FROM players WHERE elo > ?", (elo,)).fetchone()[0]


def db_day_start(now: datetime) -> datetime:
    local = now.astimezone(LOCAL_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def db_week_start(now: datetime) -> datetime:
    return db_day_start(now) - timedelta(days=now.astimezone(LOCAL_TZ).weekday())


# ---- admin helpers ----
async def db_find_players(query: str):
    if query.lstrip("-").isdigit():
        return conn.execute(
            "SELECT * FROM players WHERE tg_id = ? OR student_id = ? LIMIT 10",
            (int(query), query)).fetchall()
    return conn.execute(
        "SELECT * FROM players WHERE pylower(name) LIKE ? OR student_id = ? LIMIT 10",
        (f"%{query.lower()}%", query.upper())).fetchall()  # student ids are stored in upper case


async def db_delete_player(tg_id: int) -> bool:
    with conn:
        return conn.execute("DELETE FROM players WHERE tg_id = ?", (tg_id,)).rowcount == 1


async def db_set_elo(tg_id: int, elo: int) -> bool:
    # manual fix: intentionally NOT written to elo_history (does not affect daily/weekly tops)
    with conn:
        return conn.execute("UPDATE players SET elo = ? WHERE tg_id = ?",
                            (max(0, elo), tg_id)).rowcount == 1


async def db_stats():
    return conn.execute(
        """
        SELECT COUNT(*) AS players,
               COALESCE(SUM(wins), 0) AS battles,
               COALESCE(MAX(elo), 0) AS max_elo
        FROM players
        """).fetchone()


async def db_all_ids() -> list[int]:
    return [r["tg_id"] for r in conn.execute("SELECT tg_id FROM players").fetchall()]


# ==========================================================================
# 3. ELO
# ==========================================================================

def elo_step(score_a: float, score_b: float) -> int:
    """How many elo points the battle is worth, by the gap between the two scores.
    Gap up to 1.0 inclusive -> 25, above 1.0 -> 28."""
    gap = round(abs(score_a - score_b), 1)  # scores are shown with 1 decimal; avoid float noise
    return ELO_STEP_BIG if gap > BIG_GAP else ELO_STEP


# ==========================================================================
# 4. MATCHMAKING
# ==========================================================================


@dataclass
class Entry:
    tg_id: int
    chat_id: int
    name: str
    elo: int
    score: float
    joined: float  # time.monotonic()


class Matchmaker:
    """In-memory queue. Everything runs in one event loop, so no locks are needed."""

    def __init__(self) -> None:
        self.queue: dict[int, Entry] = {}

    def __contains__(self, tg_id: int) -> bool:
        return tg_id in self.queue

    def add(self, entry: Entry) -> None:
        self.queue[entry.tg_id] = entry

    def remove(self, tg_id: int) -> bool:
        return self.queue.pop(tg_id, None) is not None

    @staticmethod
    def search_range(entry: Entry, now: float) -> int:
        """+-200 at first, +100 every 30 seconds of waiting."""
        steps = int((now - entry.joined) // MATCH_RANGE_EVERY)
        return MATCH_RANGE + steps * MATCH_RANGE_STEP

    def pop_pairs(self, now: float) -> list[tuple[Entry, Entry]]:
        pairs = []
        free = sorted(self.queue.values(), key=lambda e: e.joined)  # longest waiting first
        used: set[int] = set()
        for a in free:
            if a.tg_id in used:
                continue
            best, best_diff = None, None
            for b in free:
                if b.tg_id == a.tg_id or b.tg_id in used:
                    continue
                diff = abs(a.elo - b.elo)
                # either player's range is enough, so a long waiter can meet a newcomer
                if diff <= max(self.search_range(a, now), self.search_range(b, now)):
                    if best_diff is None or diff < best_diff:
                        best, best_diff = b, diff
            if best:
                used.update((a.tg_id, best.tg_id))
                pairs.append((a, best))
        for a, b in pairs:
            self.queue.pop(a.tg_id, None)
            self.queue.pop(b.tg_id, None)
        return pairs

    def pop_expired(self, now: float) -> list[Entry]:
        gone = [e for e in self.queue.values() if now - e.joined > MATCH_TIMEOUT]
        for e in gone:
            self.queue.pop(e.tg_id, None)
        return gone


matchmaker = Matchmaker()


async def run_loop(on_match, on_timeout, interval: float = 2.0) -> None:
    """Background task: pairs players and reports timeouts via async callbacks."""
    while True:
        await asyncio.sleep(interval)
        try:
            now = time.monotonic()
            for a, b in matchmaker.pop_pairs(now):
                asyncio.create_task(_safe(on_match(a, b)))
            for e in matchmaker.pop_expired(now):
                asyncio.create_task(_safe(on_timeout(e)))
        except Exception:
            log.exception("matchmaking loop error")


async def _safe(coro) -> None:
    try:
        await coro
    except Exception:
        log.exception("matchmaking callback error")


# ==========================================================================
# 5. SCORING
# ==========================================================================

# Face "score" from photo proportions (MediaPipe Face Mesh + OpenCV).
# This is a home-made metric (symmetry + a few classic proportions), NOT a real PSL.
# Photos are decoded from bytes in memory only; nothing is written to disk.

# ---- thresholds (tune on real photos) ----
MIN_SIDE = 200            # px, smallest allowed image side
MIN_BRIGHTNESS = 50       # mean gray of the face crop, 0..255
MAX_BRIGHTNESS = 215
MIN_SHARPNESS = 50.0      # variance of Laplacian on the 256x256 face crop
MIN_FACE_SHARE = 0.15     # face width / image width
MAX_TURN = 0.10           # allowed head turn (nose offset from the center)

# ---- score shape (tune on real photos) ----
SYM_TOL = 0.035           # asymmetry (share of face width) that drops symmetry to ~37%
SCORE_BASE = 2.0
SCORE_RANGE = 8.0         # score = BASE + RANGE * quality, clipped to 1..10

# FaceMesh landmark indices
TOP, CHIN = 10, 152
BROW_MID, NOSE_TIP, NOSE_BASE = 9, 1, 2
FACE_L, FACE_R = 234, 454
EYE_OUT_L, EYE_IN_L, EYE_IN_R, EYE_OUT_R = 33, 133, 362, 263
NOSTRIL_L, NOSTRIL_R = 98, 327
MOUTH_L, MOUTH_R = 61, 291
MIDLINE = [10, 9, 168, 1, 2, 0, 17, 152]
SYM_PAIRS = [(33, 263), (133, 362), (159, 386), (145, 374), (61, 291), (234, 454),
             (98, 327), (70, 300), (105, 334), (172, 397)]


class PhotoError(Exception):
    """Photo rejected. The text is shown to the user (Russian)."""


def _dist(p, a, b) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def _closeness(value: float, ideal: float, tol: float) -> float:
    return max(0.0, 1.0 - abs(value - ideal) / tol)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PhotoError("Не получилось открыть фото. Отправь обычное фото (не файлом).")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise PhotoError("Фото слишком маленькое. Нужно хотя бы 200 пикселей по короткой стороне.")
    return img


def _face_landmarks(img: np.ndarray) -> np.ndarray:
    """Returns 468x2 landmarks in pixels. Raises PhotoError if there is not exactly one face."""
    import mediapipe as mp  # imported lazily: heavy, and not needed for the pure math below

    h, w = img.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=2,
        refine_landmarks=False, min_detection_confidence=0.6,
    ) as mesh:
        result = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    faces = result.multi_face_landmarks or []
    if not faces:
        gray_mean = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
        if gray_mean < MIN_BRIGHTNESS:
            raise PhotoError("Фото слишком тёмное. Включи свет и сфотографируй лицо ещё раз.")
        raise PhotoError("Не вижу лица. Сфотографируй лицо крупно, анфас, без маски и очков.")
    if len(faces) > 1:
        raise PhotoError("На фото больше одного лица. Нужно фото только с тобой.")
    return np.array([[p.x * w, p.y * h] for p in faces[0].landmark], dtype=np.float64)


def check_face_quality(img: np.ndarray, pts: np.ndarray) -> None:
    """Brightness / contrast / blur / size of the face area."""
    h, w = img.shape[:2]
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 10 or y1 - y0 < 10:
        raise PhotoError("Лицо не видно целиком. Отойди чуть дальше и держи лицо в кадре.")
    if (x1 - x0) / w < MIN_FACE_SHARE:
        raise PhotoError("Лицо слишком маленькое в кадре. Сфотографируй ближе.")

    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < MIN_BRIGHTNESS:
        raise PhotoError("Фото слишком тёмное. Включи свет и сфотографируй ещё раз.")
    if mean > MAX_BRIGHTNESS:
        raise PhotoError("Фото пересвечено. Уйди от яркого света и сфотографируй ещё раз.")

    sharp = cv2.Laplacian(cv2.resize(gray, (256, 256)), cv2.CV_64F).var()
    if sharp < MIN_SHARPNESS:
        raise PhotoError("Фото смазанное. Держи телефон ровно и сфотографируй ещё раз.")


def _level(pts: np.ndarray) -> np.ndarray:
    """Rotates the face so that its midline is vertical (removes head tilt)."""
    mid = pts[MIDLINE]
    centered = mid - mid.mean(axis=0)
    direction = np.linalg.svd(centered)[2][0]
    if direction[1] < 0:  # make it point from forehead to chin
        direction = -direction
    angle = np.arctan2(direction[0], direction[1])
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (pts - mid.mean(axis=0)) @ rot.T


def score_from_landmarks(pts: np.ndarray) -> float:
    """Pure math: 468x2 landmarks -> score 1..10. Raises PhotoError if the head is turned."""
    p = _level(pts)
    width = _dist(p, FACE_L, FACE_R)
    if width < 1e-6:
        raise PhotoError("Не получилось разобрать лицо. Попробуй другое фото.")

    # head turn: the nose tip should sit in the middle between the face edges
    turn = abs((p[NOSE_TIP][0] - p[FACE_L][0]) / (p[FACE_R][0] - p[FACE_L][0]) - 0.5)
    if turn > MAX_TURN:
        raise PhotoError("Голова повёрнута. Смотри прямо в камеру.")

    # symmetry: mirrored landmarks must be equally far from the midline and at the same height
    asym = []
    for a, b in SYM_PAIRS:
        asym.append(abs(abs(p[a][0]) - abs(p[b][0])) / width)   # distance to midline
        asym.append(abs(p[a][1] - p[b][1]) / width)             # height difference
    symmetry = float(np.exp(-(np.mean(asym) / SYM_TOL) ** 2))

    # classic proportions: (value, ideal, tolerance)
    eye_w = (_dist(p, EYE_OUT_L, EYE_IN_L) + _dist(p, EYE_IN_R, EYE_OUT_R)) / 2
    inner = _dist(p, EYE_IN_L, EYE_IN_R)
    nose_w = _dist(p, NOSTRIL_L, NOSTRIL_R)
    mouth_w = _dist(p, MOUTH_L, MOUTH_R)
    mid_face = _dist(p, BROW_MID, NOSE_BASE)
    low_face = _dist(p, NOSE_BASE, CHIN)
    proportions = [
        _closeness(inner / eye_w, 1.0, 0.35),        # eyes are one eye-width apart
        _closeness(nose_w / inner, 1.0, 0.35),       # nose width ~ distance between eyes
        _closeness(mouth_w / nose_w, 1.5, 0.5),      # mouth ~ 1.5 nose widths
        _closeness(mid_face / low_face, 1.0, 0.3),   # middle and lower thirds are equal
    ]
    quality = 0.5 * symmetry + 0.5 * float(np.mean(proportions))
    return float(np.clip(SCORE_BASE + SCORE_RANGE * quality, 1.0, 10.0))


def score_photo(data: bytes) -> float:
    """Blocking. bytes -> score of one photo, or PhotoError."""
    img = _decode(data)
    pts = _face_landmarks(img)
    check_face_quality(img, pts)
    return score_from_landmarks(pts)


async def score_photo_async(data: bytes) -> float:
    return await asyncio.to_thread(score_photo, data)


def average_score(scores: list[float]) -> float:
    """Final score of a battle = mean of the photos, 1 decimal (what the players see)."""
    return round(sum(scores) / len(scores), 1)


# ==========================================================================
# 6. UI
# ==========================================================================

_bad_photos: set[str] = set()  # keys whose picture Telegram rejected -> show text only

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
PAGE_SIZE = 20


def main_kb(is_admin: bool = False) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="⚔️ Батл", callback_data="battle")
    kb.button(text="🏆 Топ", callback_data="top")
    kb.button(text="👤 Профиль", callback_data="profile")
    kb.button(text="📜 Правила", callback_data="rules")
    if is_admin:
        kb.button(text="🛠 Админка", callback_data="adm")
    kb.adjust(2)
    return kb.as_markup()


def back_kb(data: str = "menu", text: str = "‹ Назад") -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text=text, callback_data=data)
    return kb.as_markup()


def top_menu_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📅 Топ дня", callback_data="top:day:0")
    kb.button(text="📆 Топ недели", callback_data="top:week:0")
    kb.button(text="👑 За всё время", callback_data="top:all:0")
    kb.button(text="‹ Назад", callback_data="menu")
    kb.adjust(1)
    return kb.as_markup()


def top_page_kb(kind: str, page: int, pages: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    nav = 0
    if page > 0:
        kb.button(text="‹", callback_data=f"top:{kind}:{page - 1}")
        nav += 1
    if page < pages - 1:
        kb.button(text="›", callback_data=f"top:{kind}:{page + 1}")
        nav += 1
    kb.button(text="‹ К топам", callback_data="top")
    if nav:
        kb.adjust(nav, 1)
    else:
        kb.adjust(1)
    return kb.as_markup()


def player_label(name: str, tg_id: int) -> str:
    return f"{escape(name)} · <code>{tg_id}</code>"


def format_top(title: str, rows, page: int, unit: str, value_key: str) -> tuple[str, int]:
    """Returns (text, pages). `rows` is the full top-100."""
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page = min(max(page, 0), pages - 1)
    lines = [f"<b>{title}</b>\n"]
    if not rows:
        lines.append("Пока никого нет.")
    start = page * PAGE_SIZE
    for i, r in enumerate(rows[start:start + PAGE_SIZE], start=start + 1):
        value = r[value_key]
        shown = f"+{value}" if unit == "gain" else str(value)
        lines.append(f"{MEDALS.get(i, f'{i}.')} {player_label(r['name'], r['tg_id'])} — <b>{shown}</b>")
    if pages > 1:
        lines.append(f"\nСтраница {page + 1} из {pages}")
    return "\n".join(lines), pages


async def show(event: CallbackQuery | Message, key: str | None, text: str,
               kb: InlineKeyboardMarkup | None = None) -> None:
    """Shows a screen in one window. `key` = picture from PHOTOS (None = text only).
    Falls back to plain text if the picture is missing or Telegram rejects it."""
    photo = "" if key is None or key in _bad_photos else PHOTOS.get(key, "")
    msg = event.message if isinstance(event, CallbackQuery) else None

    if msg is not None:
        try:
            if photo and msg.photo:
                await msg.edit_media(InputMediaPhoto(media=photo, caption=text), reply_markup=kb)
                return
            if not photo and not msg.photo:
                await msg.edit_text(text, reply_markup=kb)
                return
        except TelegramBadRequest as e:
            if "not modified" in str(e):
                return
            # anything else: drop the old message and send the screen again below
        try:
            await msg.delete()
        except TelegramBadRequest:
            pass
        target = msg
    else:
        target = event

    if photo:
        try:
            await target.answer_photo(photo, caption=text, reply_markup=kb)
            return
        except TelegramBadRequest:
            log.warning("Picture '%s' rejected by Telegram, using text", key)
            _bad_photos.add(key)
    await target.answer(text, reply_markup=kb)


# ==========================================================================
# 7. HANDLERS
# ==========================================================================

user_router = Router()


NAME_RE = re.compile(r"^[\w .'\-]{2,40}$")
STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9\-/]{4,20}$")
_locks: dict[int, asyncio.Lock] = {}  # one lock per user: albums arrive as parallel updates


class Reg(StatesGroup):
    name = State()
    student_id = State()


class Battle(StatesGroup):
    photos = State()
    waiting = State()


RULES_TEXT = (
    "<b>📜 Правила</b>\n\n"
    f"• Старт — <b>{START_ELO}</b> эло. Победа/поражение: ±{ELO_STEP}, "
    f"а если разрыв в оценках больше {BIG_GAP} — ±{ELO_STEP_BIG}.\n"
    "• Эло не падает ниже 0.\n"
    f"• Для батла нужно {PHOTOS_PER_BATTLE} фото лица: анфас, при хорошем свете, "
    "без очков и масок, только ты в кадре.\n"
    "• Оценка (score) считается по пропорциям и симметрии лица. Это не настоящий PSL, "
    "а наша внутренняя метрика.\n"
    "• Соперник подбирается по эло: сначала ±200, каждые 30 секунд диапазон шире на 100.\n"
    "• Фото нигде не сохраняются: бот считает оценку и сразу забывает снимок.\n"
    "• Топ дня и недели — кто сильнее поднял эло за период."
)


def _is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID


async def _try_delete(message: Message) -> None:
    try:
        await message.delete()
    except TelegramBadRequest:
        pass


async def show_menu(event: Message | CallbackQuery) -> None:
    await show(event, "main", "<b>Mog Battle</b>\nГлавное меню. Выбирай.",
               main_kb(_is_admin(event.from_user.id)))


async def _registered(cb: CallbackQuery):
    player = await db_get_player(cb.from_user.id)
    if player is None:
        await cb.answer("Сначала нажми /start и зарегистрируйся", show_alert=True)
    return player


def _cancel_kb():
    return back_kb("menu", "❌ Отмена")


# =================== START / REGISTRATION ===================
@user_router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    matchmaker.remove(message.from_user.id)
    await state.clear()
    if await db_get_player(message.from_user.id):
        await show_menu(message)
        return
    await show(message, "welcome",
               "<b>Mog Battle колледжа</b> ⚔️\n\nРегистрация. Как тебя зовут? "
               "Напиши имя и фамилию — они появятся в топе.")
    await state.set_state(Reg.name)


@user_router.message(Reg.name, F.text, ~F.text.startswith("/"))
async def reg_name(message: Message, state: FSMContext):
    name = " ".join(message.text.split())
    if not NAME_RE.match(name):
        await message.answer("Имя от 2 до 40 символов: буквы, цифры, пробел, точка, дефис. Попробуй ещё раз.")
        return
    await state.update_data(name=name)
    await state.set_state(Reg.student_id)
    await message.answer("Теперь номер студенческого (ID-карты). Он нужен только чтобы не было дублей "
                         "и нигде не показывается.")


@user_router.message(Reg.student_id, F.text, ~F.text.startswith("/"))
async def reg_student_id(message: Message, state: FSMContext):
    student_id = message.text.strip().upper()
    if not STUDENT_ID_RE.match(student_id):
        await message.answer("Номер: 4–20 символов, только латиница, цифры, «-» и «/». Попробуй ещё раз.")
        return
    data = await state.get_data()
    created = await db_create_player(message.from_user.id, student_id, data["name"])
    await _try_delete(message)  # keep the number out of the chat history
    if not created:
        await message.answer("Этот номер уже зарегистрирован. Если это твой номер — напиши админу.")
        return
    await state.clear()
    await message.answer(f"✅ Готово, {escape(data['name'])}! Ты в игре, старт — {START_ELO} эло.")
    await show_menu(message)


@user_router.message(Command("menu"))
async def cmd_menu(message: Message, state: FSMContext):
    matchmaker.remove(message.from_user.id)
    await state.clear()
    if await db_get_player(message.from_user.id) is None:
        await message.answer("Сначала нажми /start и зарегистрируйся.")
        return
    await show_menu(message)


@user_router.callback_query(F.data == "menu")
async def cb_menu(cb: CallbackQuery, state: FSMContext):
    matchmaker.remove(cb.from_user.id)  # leaving the menu while searching cancels the search
    await state.clear()
    await show_menu(cb)
    await cb.answer()


# =================== PROFILE / RULES / TOPS ===================
@user_router.callback_query(F.data == "profile")
async def cb_profile(cb: CallbackQuery):
    p = await _registered(cb)
    if p is None:
        return
    rank = await db_rank_of(p["elo"])
    text = (f"<b>👤 {escape(p['name'])}</b>\n"
            f"Telegram ID: <code>{p['tg_id']}</code>\n"
            f"Студенческий: <code>{escape(p['student_id'])}</code>\n\n"
            f"⭐ Эло: <b>{p['elo']}</b>\n"
            f"🏅 Место в общем рейтинге: <b>#{rank}</b>\n"
            f"✅ Победы: {p['wins']}   ❌ Поражения: {p['losses']}")
    await show(cb, "profile", text, back_kb())
    await cb.answer()


@user_router.callback_query(F.data == "rules")
async def cb_rules(cb: CallbackQuery):
    await show(cb, "rules", RULES_TEXT, back_kb())
    await cb.answer()


@user_router.callback_query(F.data == "top")
async def cb_top_menu(cb: CallbackQuery):
    await show(cb, "top", "<b>🏆 Топы</b>\nВыбери период.", top_menu_kb())
    await cb.answer()


@user_router.callback_query(F.data.regexp(r"^top:(day|week|all):\d+$"))
async def cb_top(cb: CallbackQuery):
    _, kind, page = cb.data.split(":")
    now = datetime.now(timezone.utc)
    if kind == "all":
        rows = await db_top_alltime(100)
        text, pages = format_top("👑 Топ за всё время", rows, int(page), "elo", "elo")
    elif kind == "day":
        rows = await db_top_period(db_day_start(now), 100)
        text, pages = format_top("📅 Топ дня (прирост эло)", rows, int(page), "gain", "gain")
    else:
        rows = await db_top_period(db_week_start(now), 100)
        text, pages = format_top("📆 Топ недели (прирост эло)", rows, int(page), "gain", "gain")
    await show(cb, None, text, top_page_kb(kind, min(int(page), pages - 1), pages))
    await cb.answer()


# =================== BATTLE ===================
def _photo_prompt(n: int) -> str:
    return (f"Пришли фото <b>{n} из {PHOTOS_PER_BATTLE}</b>.\n"
            "Лицо анфас, хороший свет, без очков и масок, только ты. Отправляй как обычное фото, не файлом.")


@user_router.callback_query(F.data == "battle")
async def cb_battle(cb: CallbackQuery, state: FSMContext):
    if await _registered(cb) is None:
        return
    if cb.from_user.id in matchmaker:
        await cb.answer("Ты уже в поиске соперника", show_alert=True)
        return
    await state.set_state(Battle.photos)
    await state.update_data(scores=[], seen=[])
    await show(cb, "battle", "<b>⚔️ Батл</b>\n\n" + _photo_prompt(1), _cancel_kb())
    await cb.answer()


@user_router.message(Battle.photos, F.photo)
async def battle_photo(message: Message, state: FSMContext):
    uid = message.from_user.id
    async with _locks.setdefault(uid, asyncio.Lock()):
        if await state.get_state() != Battle.photos.state:
            return  # extra photos from an album after we already have enough
        data = await state.get_data()
        scores: list[float] = data.get("scores", [])
        seen: list[str] = data.get("seen", [])
        n = len(scores) + 1

        photo = message.photo[-1]
        if photo.file_unique_id in seen:
            await message.answer("Это то же самое фото. Пришли другое.")
            return

        buf = io.BytesIO()
        await message.bot.download(photo, destination=buf)  # memory only, no file on disk
        try:
            score = await score_photo_async(buf.getvalue())
        except PhotoError as e:
            await message.answer(f"❌ {e}\n\nПришли фото {n} из {PHOTOS_PER_BATTLE} заново.",
                                 reply_markup=_cancel_kb())
            await _try_delete(message)
            return
        except Exception:
            log.exception("photo scoring failed")
            await message.answer("Не получилось обработать фото. Попробуй другое.", reply_markup=_cancel_kb())
            return
        finally:
            buf.close()

        scores.append(score)
        seen.append(photo.file_unique_id)
        await state.update_data(scores=scores, seen=seen)
        await _try_delete(message)

        if len(scores) < PHOTOS_PER_BATTLE:
            await message.answer(f"✅ Фото {len(scores)} принято.\n\n" + _photo_prompt(len(scores) + 1),
                                 reply_markup=_cancel_kb())
            return

        player = await db_get_player(uid)
        if player is None:
            await state.clear()
            await message.answer("Тебя нет в рейтинге. Нажми /start.")
            return
        final = average_score(scores)
        await state.set_state(Battle.waiting)
        matchmaker.add(Entry(uid, message.chat.id, player["name"], player["elo"], final, time.monotonic()))
        await message.answer(
            "✅ Все фото приняты.\n🔎 Ищу соперника по эло (±200, каждые 30 секунд шире на 100). "
            "Результат пришлю сюда.",
            reply_markup=back_kb("menu", "❌ Отменить поиск"))


@user_router.message(Battle.photos)
async def battle_not_photo(message: Message):
    await message.answer("Нужно именно фото (не файлом и не текстом).", reply_markup=_cancel_kb())


@user_router.message(Battle.waiting)
async def battle_waiting(message: Message):
    await message.answer("Ищу соперника… Подожди или нажми «Отменить поиск» выше.")


def _after_battle_kb():
    kb = InlineKeyboardBuilder()
    kb.button(text="⚔️ Ещё батл", callback_data="battle")
    kb.button(text="‹ Меню", callback_data="menu")
    kb.adjust(1)
    return kb.as_markup()


def make_callbacks(bot: Bot, storage):
    """Callbacks for the matchmaking loop (they need the bot and the FSM storage)."""

    async def release(uid: int) -> None:
        key = StorageKey(bot_id=bot.id, chat_id=uid, user_id=uid)
        await FSMContext(storage=storage, key=key).clear()

    async def send(entry: Entry, text: str) -> None:
        await bot.send_message(entry.chat_id, text, reply_markup=_after_battle_kb())

    async def on_match(a: Entry, b: Entry) -> None:
        await release(a.tg_id)
        await release(b.tg_id)

        if a.score == b.score:
            for me, other in ((a, b), (b, a)):
                await _quiet(send(me, f"⚔️ <b>Батл завершён</b>\n\nТы: <b>{me.score}</b>\n"
                                      f"Соперник: {player_label(other.name, other.tg_id)} — <b>{other.score}</b>\n\n"
                                      "🤝 Ничья, эло не меняется."))
            return

        step = elo_step(a.score, b.score)
        win, lose = (a, b) if a.score > b.score else (b, a)
        result = await db_apply_result(win.tg_id, lose.tg_id, step, step)
        if result is None:
            for e in (a, b):
                await _quiet(send(e, "Соперник вышел из рейтинга, батл отменён."))
            return
        new_win, new_lose, real_lose = result

        head = "⚔️ <b>Батл завершён</b>\n\n"
        await _quiet(send(win, head + f"Ты: <b>{win.score}</b>\n"
                                      f"Соперник: {player_label(lose.name, lose.tg_id)} — <b>{lose.score}</b>\n\n"
                                      f"🏆 Победа! <b>+{step}</b> эло → <b>{new_win}</b>"))
        await _quiet(send(lose, head + f"Ты: <b>{lose.score}</b>\n"
                                       f"Соперник: {player_label(win.name, win.tg_id)} — <b>{win.score}</b>\n\n"
                                       f"💀 Поражение. <b>{real_lose}</b> эло → <b>{new_lose}</b>"))

    async def on_timeout(e: Entry) -> None:
        await release(e.tg_id)
        await _quiet(send(e, "😕 За 5 минут соперника в твоём диапазоне не нашлось. "
                             "Попробуй позже, когда в боте будет больше людей."))

    return on_match, on_timeout


async def _quiet(coro) -> None:
    """One failed message (e.g. the user blocked the bot) must not break the rest."""
    try:
        await coro
    except Exception:
        log.exception("failed to send battle message")


# =================== FALLBACK ===================
@user_router.message(StateFilter(None))
async def fallback(message: Message):
    await message.answer("Открой меню: /menu")


# ==========================================================================
# 8. ADMIN
# ==========================================================================

admin_router = Router()
# every handler in this router works only for the single admin
admin_router.message.filter(F.from_user.id == ADMIN_ID)
admin_router.callback_query.filter(F.from_user.id == ADMIN_ID)


class Adm(StatesGroup):
    find = State()
    delete = State()
    set_elo = State()
    broadcast = State()


def panel_kb():
    kb = InlineKeyboardBuilder()
    kb.button(text="📊 Статистика", callback_data="adm_stats")
    kb.button(text="🔎 Найти игрока", callback_data="adm_find")
    kb.button(text="🗑 Удалить игрока", callback_data="adm_del")
    kb.button(text="✏️ Изменить эло", callback_data="adm_elo")
    kb.button(text="📢 Рассылка", callback_data="adm_bc")
    kb.button(text="‹ В меню", callback_data="menu")
    kb.adjust(2)
    return kb.as_markup()


def _panel_back():
    return back_kb("adm", "‹ В админку")


def _row(p) -> str:
    return (f"<b>{escape(p['name'])}</b> · ID <code>{p['tg_id']}</code>\n"
            f"студенческий <code>{escape(p['student_id'])}</code> · эло {p['elo']} · "
            f"{p['wins']}W/{p['losses']}L")


async def _prompt(cb: CallbackQuery, state: FSMContext, new_state: State, text: str) -> None:
    await state.set_state(new_state)
    await show(cb, "admin", text, _panel_back())
    await cb.answer()


@admin_router.message(Command("admin"))
async def cmd_admin(message: Message, state: FSMContext):
    await state.clear()
    await show(message, "admin", "<b>🛠 Админка</b>", panel_kb())


@admin_router.callback_query(F.data == "adm")
async def cb_panel(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await show(cb, "admin", "<b>🛠 Админка</b>", panel_kb())
    await cb.answer()


@admin_router.callback_query(F.data == "adm_stats")
async def cb_stats(cb: CallbackQuery):
    s = await db_stats()
    text = (f"<b>📊 Статистика</b>\n\nИгроков: <b>{s['players']}</b>\n"
            f"Батлов сыграно: <b>{s['battles']}</b>\n"
            f"Максимальное эло: <b>{s['max_elo']}</b>\n"
            f"Сейчас в поиске: <b>{len(matchmaker.queue)}</b>")
    await show(cb, "admin", text, _panel_back())
    await cb.answer()


# ---- find ----
@admin_router.callback_query(F.data == "adm_find")
async def cb_find(cb: CallbackQuery, state: FSMContext):
    await _prompt(cb, state, Adm.find, "Пришли Telegram ID, студенческий номер или часть имени.")


@admin_router.message(Adm.find, F.text, ~F.text.startswith("/"))
async def find_input(message: Message, state: FSMContext):
    rows = await db_find_players(message.text.strip())
    text = "\n\n".join(_row(p) for p in rows) if rows else "Никого не нашёл."
    await state.clear()
    await message.answer(text, reply_markup=panel_kb())


# ---- delete ----
@admin_router.callback_query(F.data == "adm_del")
async def cb_del(cb: CallbackQuery, state: FSMContext):
    await _prompt(cb, state, Adm.delete, "Пришли Telegram ID игрока, которого удалить.")


@admin_router.message(Adm.delete, F.text, ~F.text.startswith("/"))
async def del_input(message: Message, state: FSMContext):
    raw = message.text.strip()
    player = await db_get_player(int(raw)) if raw.isdigit() else None
    if player is None:
        await message.answer("Игрока с таким Telegram ID нет. Пришли ID ещё раз.", reply_markup=_panel_back())
        return
    await state.clear()
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, удалить", callback_data=f"adm_del_yes:{player['tg_id']}")
    kb.button(text="Отмена", callback_data="adm")
    kb.adjust(1)
    await message.answer("Удалить игрока?\n\n" + _row(player) + "\n\nИстория эло тоже удалится.",
                         reply_markup=kb.as_markup())


@admin_router.callback_query(F.data.regexp(r"^adm_del_yes:\d+$"))
async def cb_del_yes(cb: CallbackQuery):
    tg_id = int(cb.data.split(":")[1])
    matchmaker.remove(tg_id)
    ok = await db_delete_player(tg_id)
    await show(cb, "admin", "🗑 Удалён." if ok else "Игрок уже удалён.", panel_kb())
    await cb.answer()


# ---- manual elo ----
@admin_router.callback_query(F.data == "adm_elo")
async def cb_elo(cb: CallbackQuery, state: FSMContext):
    await _prompt(cb, state, Adm.set_elo, "Пришли «Telegram ID и новое эло», например: <code>123456789 1200</code>")


@admin_router.message(Adm.set_elo, F.text, ~F.text.startswith("/"))
async def elo_input(message: Message, state: FSMContext):
    parts = message.text.split()
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        await message.answer("Формат: <code>ID число</code>, например <code>123456789 1200</code>",
                             reply_markup=_panel_back())
        return
    tg_id, elo = int(parts[0]), int(parts[1])
    await state.clear()
    if not await db_set_elo(tg_id, elo):
        await message.answer("Игрока с таким ID нет.", reply_markup=panel_kb())
        return
    if tg_id in matchmaker.queue:
        matchmaker.queue[tg_id].elo = elo  # keep matchmaking in sync
    await message.answer(f"✅ Эло игрока <code>{tg_id}</code> теперь {elo}.", reply_markup=panel_kb())


# ---- broadcast ----
@admin_router.callback_query(F.data == "adm_bc")
async def cb_bc(cb: CallbackQuery, state: FSMContext):
    await _prompt(cb, state, Adm.broadcast, "Пришли текст рассылки (получат все зарегистрированные игроки).")


@admin_router.message(Adm.broadcast, F.text, ~F.text.startswith("/"))
async def bc_input(message: Message, state: FSMContext):
    await state.update_data(text=message.text)
    kb = InlineKeyboardBuilder()
    kb.button(text="📢 Отправить всем", callback_data="adm_bc_yes")
    kb.button(text="Отмена", callback_data="adm")
    kb.adjust(1)
    await message.answer("Предпросмотр:\n\n" + escape(message.text), reply_markup=kb.as_markup())


@admin_router.callback_query(F.data == "adm_bc_yes", Adm.broadcast)
async def cb_bc_yes(cb: CallbackQuery, state: FSMContext):
    text = (await state.get_data()).get("text")
    await state.clear()
    if not text:
        await cb.answer("Текста нет", show_alert=True)
        return
    await cb.answer("Отправляю…")
    ids = await db_all_ids()
    sent = 0
    for uid in ids:
        try:
            await cb.bot.send_message(uid, escape(text))
            sent += 1
        except Exception:  # blocked the bot, deleted chat, etc.
            log.info("broadcast: cannot deliver to %s", uid)
        await asyncio.sleep(0.05)  # stay under Telegram's flood limits
    await cb.message.answer(f"📢 Доставлено {sent} из {len(ids)}.", reply_markup=panel_kb())


# ==========================================================================
# 9. MAIN
# ==========================================================================

async def main() -> None:
    if BOT_TOKEN.startswith("8808641162:AAE4-Pzv6M6LXsVPO2w4kWqv2CDvqGiYELU"):
        raise SystemExit("Set the BOT_TOKEN environment variable (token from @BotFather).")

    await db_init()

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(admin_router)     # admin first: its FSM states must win over user handlers
    dp.include_router(user_router)

    on_match, on_timeout = make_callbacks(bot, storage)
    matcher = asyncio.create_task(run_loop(on_match, on_timeout))

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        matcher.cancel()
        await db_close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
