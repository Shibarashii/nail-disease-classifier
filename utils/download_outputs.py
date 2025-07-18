from huggingface_hub import snapshot_download
from pathlib import Path
from constants import HUGGING_FACE_ID, HUGGING_FACE_REPO


project_root = Path(__file__).resolve().parent.parent
target_dir = project_root / "outputs"
patterns = "models/**"

snapshot_download(
    repo_id=f"{HUGGING_FACE_ID}/{HUGGING_FACE_REPO}",
    allow_patterns=patterns,
    local_dir=target_dir)

print(f"Downloaded finished. Located at: {target_dir}")
