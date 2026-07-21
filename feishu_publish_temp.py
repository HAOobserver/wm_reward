import json
import pathlib
import re
import sys
import time

import requests


ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = pathlib.Path.home() / ".cursor" / "mcp.json"
DRAFT_PATH = ROOT / "feishu_paper_talk_draft.md"
TITLE = "四篇 VLA/WAM 论文精讲：VLA-RFT、WAM-RL、LaWAM、VLA-OPD"
EXISTING_DOC_ID = "WUgndaafyowiPLx5hL5ceDVInVe"
EXISTING_DOC_URL = "https://www.feishu.cn/docx/WUgndaafyowiPLx5hL5ceDVInVe"


def parse_rpc_response(response: requests.Response) -> dict:
    response.raise_for_status()
    response.encoding = "utf-8"
    for line in response.text.splitlines():
        if line.startswith("data:"):
            return json.loads(line.removeprefix("data:").strip())
    return response.json()


def tool_call(endpoint: str, request_id: int, name: str, arguments: dict) -> tuple[dict, str]:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Accept": "application/json, text/event-stream"},
        timeout=180,
    )
    rpc = parse_rpc_response(response)
    if "error" in rpc:
        raise RuntimeError(json.dumps(rpc["error"], ensure_ascii=False))

    result = rpc.get("result", {})
    text_parts = [
        item.get("text", "")
        for item in result.get("content", [])
        if item.get("type") == "text"
    ]
    text = "\n".join(text_parts)
    if result.get("isError"):
        raise RuntimeError(text or "Feishu MCP tool returned an error")
    return result, text


def parse_tool_data(text: str) -> dict:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            if isinstance(value.get("result"), dict):
                return value["result"]
            return value
    except json.JSONDecodeError:
        pass

    data = {}
    for key in ("doc_id", "doc_url", "task_id"):
        quoted = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
        plain = re.search(rf"{key}\s*[:：]\s*(\S+)", text)
        match = quoted or plain
        if match:
            data[key] = match.group(1).rstrip(",")
    return data


def wait_for_task(endpoint: str, request_id: int, tool_name: str, task_id: str, mode: str | None = None) -> tuple[dict, str, int]:
    for _ in range(60):
        time.sleep(2)
        arguments = {"task_id": task_id}
        if mode is not None:
            arguments["mode"] = mode
        result, text = tool_call(endpoint, request_id, tool_name, arguments)
        request_id += 1
        data = parse_tool_data(text)
        if data.get("doc_id") or "成功" in text or "completed" in text.lower():
            return result, text, request_id
        if any(word in text.lower() for word in ("failed", "error")) or "失败" in text:
            raise RuntimeError(text)
    raise TimeoutError(f"Timed out waiting for {tool_name} task {task_id}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    endpoint = config["mcpServers"]["feishu-mcp"]["url"]
    chunks = DRAFT_PATH.read_text(encoding="utf-8").split("\n<!-- CHUNK -->\n")

    request_id = 100
    if EXISTING_DOC_ID:
        doc_id = EXISTING_DOC_ID
        doc_url = EXISTING_DOC_URL
        print(f"RESUMING doc_id={doc_id}")
        _, image_text = tool_call(
            endpoint,
            request_id,
            "update-doc",
            {
                "doc_id": doc_id,
                "mode": "insert_after",
                "selection_with_ellipsis": "### 3.2 原论文框架图",
                "markdown": '<image url="https://i0.wp.com/vla-rft.github.io/static/images/Figure1.png?resize=1200%2C628&ssl=1" width="1000" align="center" caption="VLA-RFT 原论文 Figure 1：用世界模型承接多组 VLA 动作 rollout，并以轨迹奖励进行 GRPO 后训练。"/>',
            },
        )
        request_id += 1
        if "warning" in image_text.lower() or "警告" in image_text:
            print(image_text)
    else:
        _, create_text = tool_call(
            endpoint,
            request_id,
            "create-doc",
            {"title": TITLE, "markdown": chunks[0]},
        )
        request_id += 1
        create_data = parse_tool_data(create_text)
        if create_data.get("task_id") and not create_data.get("doc_id"):
            _, create_text, request_id = wait_for_task(
                endpoint,
                request_id,
                "create-doc",
                create_data["task_id"],
            )
            create_data.update(parse_tool_data(create_text))

        doc_id = create_data.get("doc_id")
        doc_url = create_data.get("doc_url")
        if not doc_id:
            raise RuntimeError(f"Could not parse doc_id from create-doc response:\n{create_text}")

        print(f"CREATED doc_id={doc_id}")
        if doc_url:
            print(f"DOC_URL {doc_url}")

    for index, chunk in enumerate(chunks[1:], start=2):
        _, update_text = tool_call(
            endpoint,
            request_id,
            "update-doc",
            {"doc_id": doc_id, "mode": "append", "markdown": chunk},
        )
        request_id += 1
        update_data = parse_tool_data(update_text)
        if update_data.get("task_id"):
            _, update_text, request_id = wait_for_task(
                endpoint,
                request_id,
                "update-doc",
                update_data["task_id"],
                mode="append",
            )
        print(f"APPENDED chunk={index}")
        if "warning" in update_text.lower() or "警告" in update_text:
            print(update_text)

    print(f"COMPLETE doc_id={doc_id}")
    if doc_url:
        print(f"DOC_URL {doc_url}")


if __name__ == "__main__":
    main()
