from __future__ import annotations

import os
import re
import json
import asyncio
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import aiohttp
import aiofiles
from aiohttp import ClientTimeout
from tqdm.asyncio import tqdm

# Configurations
CSV_PATH        = "MovieGenre.csv"
IMAGE_DIR       = Path("poster_images")
MAPPING_CSV     = "poster_image_scores.csv"
FAILED_JSONL    = "poster_download_failures.jsonl"
CONCURRENCY     = 500
REQUEST_TIMEOUT = 10

SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_filename(name: str, max_len: int = 100) -> str:
    """Sanitize the filename by replacing unsafe characters and clipping length."""
    return SAFE_CHARS.sub("_", name)[:max_len] + ".jpg"

async def log_failure(title: str, url: str, reason: str, tb: str | None = None) -> None:
    """Log a failed download attempt."""
    record = {
        "title": title,
        "url": url,
        "reason": reason,
        "traceback": tb,
    }
    async with aiofiles.open(FAILED_JSONL, "a", encoding="utf-8") as f:
        await f.write(json.dumps(record, ensure_ascii=False) + "\n")

async def fetch_image(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    title: str,
    url: str,
    score: float
) -> Optional[Tuple[str, float]]:
    """Download one poster image. Return (path, score) if successful, else None."""
    if pd.isna(url) or not url.startswith("http"):
        await log_failure(title, url, "invalid_or_missing_url")
        return None

    filename  = sanitize_filename(title)
    imagepath = IMAGE_DIR / filename

    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await log_failure(title, url, f"http_status_{resp.status}")
                    return None

                content = await resp.read()
                async with aiofiles.open(imagepath, "wb") as f:
                    await f.write(content)

            return str(imagepath), score

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await log_failure(title, url, type(e).__name__)

        except Exception as e:
            await log_failure(
                title, url, "unexpected_" + type(e).__name__,
                tb=traceback.format_exc()
            )

        return None

async def scrape_all(df: pd.DataFrame) -> List[Tuple[str, float]]:
    """Download all posters concurrently. Return list of successful (path, score)."""
    timeout   = ClientTimeout(
        total=None,
        sock_read=REQUEST_TIMEOUT,
        sock_connect=REQUEST_TIMEOUT
    )
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        sem   = asyncio.Semaphore(CONCURRENCY)
        tasks = [
            fetch_image(session, sem, row["Title"], row["Poster"], row["IMDB Score"])
            for _, row in df.iterrows()
        ]

        results = [r for r in await tqdm.gather(*tasks) if r is not None]
        return results

def main() -> None:
    df = pd.read_csv(CSV_PATH, encoding="latin1")

    print(f"Starting download of {len(df):,} posters "
          f"with up to {CONCURRENCY} concurrent requests…")

    results = asyncio.run(scrape_all(df))

    pd.DataFrame(results, columns=["image_path", "imdb_score"]) \
      .to_csv(MAPPING_CSV, index=False)

    print(f"\nDownloaded {len(results):,} posters.")
    print(f"Success mapping saved → {MAPPING_CSV}")
    print(f"Failures logged (if any)   → {FAILED_JSONL}")

if __name__ == "__main__":
    main()
