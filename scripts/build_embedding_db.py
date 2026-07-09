from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triplet_landmark.build_embedding_db import main


if __name__ == "__main__":
    main()
