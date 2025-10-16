from pathlib import Path
from pydub import AudioSegment
import os, subprocess

def find_ffbin(start: Path) -> Path:
    # 1) search recursively under current tree
    cand = next(start.rglob("bin/ffmpeg.exe"), None)
    if cand: return cand.parent
    # 2) walk up parents and probe common layout ffmpeg-*/bin
    cur = start
    while True:
        cand = next(cur.glob("ffmpeg-*/bin/ffmpeg.exe"), None)
        if cand: return cand.parent
        if cur == cur.parent: break
        cur = cur.parent
    raise FileNotFoundError(f"ffmpeg.exe not found starting at {start}. "
                            f"Place an extracted ffmpeg folder (e.g., ffmpeg-8.0-essentials_build) in your project.")

ROOT  = Path.cwd()
FFBIN = find_ffbin(ROOT)


# make exes and DLLs visible to this process
os.environ["PATH"] = str(FFBIN) + ";" + os.environ.get("PATH","")
try: os.add_dll_directory(str(FFBIN))
except Exception: pass

# configure pydub
AudioSegment.converter = str(FFBIN / "ffmpeg.exe")
AudioSegment.ffprobe   = str(FFBIN / "ffprobe.exe")

# verify
subprocess.run([str(FFBIN/"ffprobe.exe"), "-version"], check=True)
print("FFBIN =", FFBIN)


def FFBIN_path() -> Path:
    """Get the path to the ffmpeg binary directory."""
    return FFBIN