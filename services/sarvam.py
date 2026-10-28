import os
import time
import requests


class SarvamError(RuntimeError):
    pass


class SarvamClient:

    BASE_URL = "https://api.sarvam.ai/v1/chat/completions"

    def __init__(self, api_key=None):

        self.api_key = api_key or os.getenv("SARVAM_API_KEY")

        self.model = os.getenv(
            "SARVAM_MODEL",
            "sarvam-105b"
        )

        # ---------------------------------------------------------
        # Main reasoning configuration
        # ---------------------------------------------------------

        raw_reasoning = os.getenv(
            "SARVAM_REASONING_EFFORT",
            "none"
        ).strip().lower()

        if raw_reasoning in {
            "",
            "none",
            "null",
            "disabled",
            "off"
        }:
            self.reasoning_effort = None

        elif raw_reasoning in {
            "low",
            "medium",
            "high"
        }:
            self.reasoning_effort = raw_reasoning

        else:
            raise SarvamError(
                "Invalid SARVAM_REASONING_EFFORT. "
                "Use low, medium, high, or none."
            )

        # ---------------------------------------------------------
        # Connection-test reasoning configuration
        # ---------------------------------------------------------

        raw_connection_reasoning = os.getenv(
            "SARVAM_CONNECTION_REASONING_EFFORT",
            "none"
        ).strip().lower()

        if raw_connection_reasoning in {
            "",
            "none",
            "null",
            "disabled",
            "off"
        }:
            self.connection_reasoning_effort = None

        elif raw_connection_reasoning in {
            "low",
            "medium",
            "high"
        }:
            self.connection_reasoning_effort = raw_connection_reasoning

        else:
            raise SarvamError(
                "Invalid SARVAM_CONNECTION_REASONING_EFFORT. "
                "Use low, medium, high, or none."
            )

        # ---------------------------------------------------------
        # API key validation
        # ---------------------------------------------------------

        if (
            not self.api_key
            or self.api_key.startswith("replace_")
        ):
            raise SarvamError(
                "SARVAM_API_KEY is not configured. "
                "Put a valid key in .env."
            )

    # =============================================================
    # CONNECTION TEST
    # =============================================================

    def connection_test(self):

        """
        Lightweight Sarvam connectivity test.

        This deliberately uses a small request and disables
        reasoning so that the test does not consume the token
        budget on hidden reasoning.
        """

        original_reasoning = self.reasoning_effort

        try:

            self.reasoning_effort = self.connection_reasoning_effort

            result = self.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Reply with exactly: "
                            "AIKOSH DOST connection successful."
                        )
                    }
                ],
                max_tokens=128,
                temperature=0
            )

            return {
                "ok": True,
                "message": result["content"],
                "usage": result.get("usage", {}),
                "model": self.model
            }

        finally:

            self.reasoning_effort = original_reasoning

    # =============================================================
    # CHAT
    # =============================================================

    def chat(
        self,
        messages,
        max_tokens=6000,
        temperature=0,
        response_format=None,
        retries=2
    ):

        if (
            not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(
                "messages must be a non-empty list"
            )

        clean_messages = []

        for i, message in enumerate(messages):

            if not isinstance(message, dict):
                raise ValueError(
                    f"messages[{i}] must be an object"
                )

            role = message.get("role")
            content = message.get("content")

            if role not in {
                "system",
                "user",
                "assistant",
                "tool"
            }:
                raise ValueError(
                    f"messages[{i}].role is invalid"
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"messages[{i}].content must be a string"
                )

            clean_messages.append(
                {
                    "role": role,
                    "content": content
                }
            )

        payload = {
            "model": self.model,
            "messages": clean_messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "reasoning_effort": self.reasoning_effort
        }

        if response_format:
            payload["response_format"] = response_format

        last_error = None

        for attempt in range(retries + 1):

            try:

                response = requests.post(
                    self.BASE_URL,
                    headers={
                        "api-subscription-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=240
                )

                # -------------------------------------------------
                # Retry transient errors
                # -------------------------------------------------

                if (
                    response.status_code
                    in {429, 500, 502, 503, 504}
                    and attempt < retries
                ):

                    time.sleep(2 ** attempt)
                    continue

                # -------------------------------------------------
                # API errors
                # -------------------------------------------------

                if response.status_code >= 400:

                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text

                    raise SarvamError(
                        f"Sarvam API error "
                        f"{response.status_code}: {detail}"
                    )

                # -------------------------------------------------
                # Parse response
                # -------------------------------------------------

                data = response.json()

                choices = data.get("choices") or []

                if not choices:
                    raise SarvamError(
                        f"Sarvam returned no choices: {data}"
                    )

                choice = choices[0]

                message = choice.get("message") or {}

                content = message.get("content")

                finish_reason = choice.get(
                    "finish_reason"
                )

                # -------------------------------------------------
                # Empty final answer
                # -------------------------------------------------

                if (
                    not isinstance(content, str)
                    or not content.strip()
                ):

                    reasoning = message.get(
                        "reasoning_content"
                    )

                    if (
                        finish_reason == "length"
                        and reasoning
                    ):

                        raise SarvamError(
                            "Sarvam returned no final answer "
                            "because the output budget was consumed "
                            "by reasoning. Reduce the prompt or "
                            "increase max_tokens."
                        )

                    raise SarvamError(
                        "Sarvam returned no final answer. "
                        f"finish_reason={finish_reason}"
                    )

                return {
                    "content": content,
                    "usage": data.get(
                        "usage",
                        {}
                    ),
                    "raw": data
                }

            except requests.RequestException as exc:

                last_error = exc

                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue

                raise SarvamError(
                    f"Network error calling Sarvam: {exc}"
                ) from exc

        raise SarvamError(
            str(
                last_error
                or "Sarvam request failed"
            )
        )

    # =============================================================
    # JSON CHAT
    # =============================================================

    def json_chat(
        self,
        system_prompt,
        user_prompt,
        schema=None,
        max_tokens=7000
    ):

        response_format = {
            "type": "json_object"
        }

        if schema:

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "aikosh_extraction",
                    "description": (
                        "Structured extraction result "
                        "for an unstructured document"
                    ),
                    "strict": True,
                    "schema": schema
                }
            }

        return self.chat(
            [
                {
                    "role": "system",
                    "content": str(system_prompt)
                },
                {
                    "role": "user",
                    "content": str(user_prompt)
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            response_format=response_format
        )

    # =============================================================
    # DOCUMENT AI / OCR
    # =============================================================

    def document_markdown(
        self,
        path,
        language="en-IN"
    ):

        try:

            from sarvamai import SarvamAI

        except ImportError as exc:

            raise SarvamError(
                "sarvamai is required for OCR. "
                "Run: pip install -U sarvamai"
            ) from exc

        client = SarvamAI(
            api_subscription_key=self.api_key
        )

        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg"
        }

        mime = mime_types.get(
            path.suffix.lower(),
            "application/octet-stream"
        )

        with open(path, "rb") as file:

            job = client.doc_ai.parse(
                file=[
                    (
                        path.name,
                        file,
                        mime
                    )
                ],
                language=language,
                output_format="md"
            )

        terminal_states = {
            "completed",
            "partially_completed",
            "failed",
            "rejected"
        }

        while True:

            status = client.doc_ai.get_status(
                job_id=job.job_id
            )

            current_status = str(
                status.status
            ).lower()

            if current_status in terminal_states:
                break

            time.sleep(3)

        if current_status not in {
            "completed",
            "partially_completed"
        }:

            raise SarvamError(
                "Sarvam Document AI OCR failed: "
                f"{status.status}"
            )

        download = client.doc_ai.get_download_url(
            job_id=job.job_id
        )

        import io
        import zipfile

        response = requests.get(
            download.url,
            timeout=120
        )

        response.raise_for_status()

        with zipfile.ZipFile(
            io.BytesIO(response.content)
        ) as archive:

            markdown_files = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".md")
            ]

            if not markdown_files:

                raise SarvamError(
                    "Sarvam Document AI completed but "
                    "returned no Markdown file."
                )

            markdown = archive.read(
                markdown_files[0]
            ).decode(
                "utf-8",
                errors="replace"
            )

        return markdown, {
            "job_id": job.job_id,
            "status": status.status,
            "provider": "Sarvam Document AI",
            "output_format": "md"
        }