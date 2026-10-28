import os
import time
import uuid
import traceback

from pathlib import Path
from datetime import datetime, timezone

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file
)

from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from services.pipeline import run_pipeline
from services.sarvam import SarvamClient


# ================================================================
# CONFIGURATION
# ================================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# FLASK
# ================================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    50 * 1024 * 1024
)


ALLOWED = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg"
}


# ================================================================
# HOME
# ================================================================

@app.get("/")
def index():

    return render_template(
        "index.html"
    )


# ================================================================
# HEALTH
# ================================================================

@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "AIKOSH DOST Phase 1",
        "sarvam_model": os.getenv(
            "SARVAM_MODEL",
            "sarvam-105b"
        ),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    })


# ================================================================
# SARVAM CONNECTION TEST
# ================================================================

@app.post("/api/test-sarvam")
def test_sarvam():

    try:

        client = SarvamClient()

        result = client.connection_test()

        return jsonify({
            "ok": True,
            "message": result["message"],
            "usage": result.get(
                "usage",
                {}
            ),
            "model": result.get(
                "model"
            )
        })

    except Exception as exc:

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 400


# ================================================================
# RUN PIPELINE
# ================================================================

@app.post("/api/run")
def run():

    started = time.time()

    run_id = uuid.uuid4().hex[:12]

    try:

        uploaded = request.files.get(
            "file"
        )

        if (
            not uploaded
            or not uploaded.filename
        ):

            return jsonify({
                "ok": False,
                "error": "No file uploaded."
            }), 400

        extension = Path(
            uploaded.filename
        ).suffix.lower()

        if extension not in ALLOWED:

            return jsonify({
                "ok": False,
                "error": (
                    "Unsupported file type. "
                    "Allowed: "
                    + ", ".join(
                        sorted(ALLOWED)
                    )
                )
            }), 400

        mode = request.form.get(
            "mode",
            "rag"
        ).lower()

        if mode not in {
            "rag",
            "direct"
        }:

            mode = "rag"

        use_case = request.form.get(
            "use_case",
            ""
        ).strip()

        if not use_case:

            return jsonify({
                "ok": False,
                "error": (
                    "Use case is required. "
                    "Describe what the final "
                    "structured dataset will be used for."
                )
            }), 400

        safe_filename = secure_filename(
            uploaded.filename
        )

        input_path = (
            UPLOAD_DIR
            / f"{run_id}_{safe_filename}"
        )

        uploaded.save(
            input_path
        )

        result = run_pipeline(
            input_path,
            run_id,
            mode,
            use_case
        )

        result["runtime_seconds"] = round(
            time.time() - started,
            2
        )

        result["ok"] = True

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "ok": False,
            "error": str(exc),
            "run_id": run_id,
            "trace": traceback.format_exc(
                limit=8
            )
        }), 500


# ================================================================
# ARTIFACT DOWNLOADS
# ================================================================

@app.get(
    "/api/download/<run_id>/<artifact>"
)
def download(
    run_id,
    artifact
):

    output_directory = (
        OUTPUT_DIR / run_id
    )

    allowed = {

        "source":
            output_directory
            / "source_preview.txt",

        "markdown":
            output_directory
            / "normalized.md",

        "chunks":
            output_directory
            / "chunks.json",

        "chunk-quality":
            output_directory
            / "chunk_quality.json",

        "embedding":
            output_directory
            / "embedding_preview.json",

        "json":
            output_directory
            / "structured.json",

        "csv":
            output_directory
            / "structured.csv",

        "xlsx":
            output_directory
            / "structured.xlsx",

        "report":
            output_directory
            / "quality_report.json",

        "provenance":
            output_directory
            / "provenance.json",

        "retrieval":
            output_directory
            / "retrieval.json"
    }

    file_path = allowed.get(
        artifact
    )

    if (
        not file_path
        or not file_path.exists()
    ):

        return jsonify({
            "error": "Artifact not found."
        }), 404

    return send_file(
        file_path,
        as_attachment=True
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )