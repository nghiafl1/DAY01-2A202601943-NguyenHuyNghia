"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency_seconds = time.perf_counter() - start_time
    response_text = response.choices[0].message.content
    return response_text, latency_seconds
    raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    raise NotImplementedError("Implement call_openai_mini")


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)
    gpt4o_cost_estimate = (
        (len(gpt4o_response.split()) / 0.75)
        / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }
    # raise NotImplementedError("Implement compare_models")
    


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt và user prompt.
    """
    # Import bên trong hàm theo yêu cầu của bài test
    from openai import OpenAI
    import time
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    latency_seconds = max(time.time() - start_time, 0.001)
    response_text = response.choices[0].message.content
    
    return response_text, latency_seconds
    raise NotImplementedError("Implement chat_with_system_prompt")


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.
    Nếu lỗi, dùng phương pháp ước lượng: max(1, len(text) // 4).
    """
    try:
        import tiktoken
        # Lấy bộ mã hoá chuẩn cho model
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback khi model lạ (ví dụ model của NVIDIA, Gemini...) hoặc lỗi mạng
        return max(1, len(text) // 4)
    raise NotImplementedError("Implement count_tokens")


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT và bảng giá.
    """
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    
    # Lấy bảng giá của model. Nếu model không có trong từ điển thì lấy giá của gpt-4o
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS.get("gpt-4o"))
    
    # Tính chi phí (giá quy định là cho 1000 token)
    input_cost = (input_tokens / 1000.0) * pricing["input"]
    output_cost = (output_tokens / 1000.0) * pricing["output"]
    total_cost = input_cost + output_cost
    
    # Trả về dict có đủ 5 key theo yêu cầu của bài test
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
    raise NotImplementedError("Implement estimate_cost")


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )

        reply_parts = []
        print("Assistant: ", end="")
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        history.append({"role": "assistant", "content": "".join(reply_parts)})
        history = history[-6:]
    # raise NotImplementedError("Implement streaming_chatbot")


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2**attempt)
    raise NotImplementedError("Implement retry_with_backoff")


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    import os
    from openai import OpenAI

    # Khởi tạo client bên trong hàm theo yêu cầu
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Nếu get_input là None, mặc định sử dụng hàm input của Python
    if get_input is None:
        get_input = input

    history = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        # Kiểm tra max_turns trước khi nhận input
        if max_turns is not None and num_turns >= max_turns:
            break

        # Đọc input của user
        user_msg = get_input()
        
        # Thoát nếu gõ quit hoặc exit (không phân biệt hoa thường)
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        # Xây dựng danh sách messages (System + History + User)
        messages = [{"role": "system", "content": persona}] + history + [{"role": "user", "content": user_msg}]

        # Bọc hàm gọi API trong lambda để truyền vào retry_with_backoff
        api_call = lambda: client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True
        )

        # Gọi API có cơ chế retry chống lỗi mạng/rate limit
        stream = retry_with_backoff(api_call)

        # Xử lý streaming và ghép chunk
        reply_parts = []
        print("Assistant: ", end="")
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        assistant_msg = "".join(reply_parts)

        # Thêm vào history và chỉ giữ lại 3 lượt gần nhất (6 messages)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        history = history[-6:]

        # Cập nhật số lượt
        num_turns += 1

        # Tính toán token và chi phí
        # Để count_tokens chạy chính xác, ta nối nội dung prompt lại thành chuỗi
        prompt_text = persona + "\n" + "\n".join([m["content"] for m in messages[1:]])
        
        cost_info = estimate_cost(prompt=prompt_text, response=assistant_msg, model=OPENAI_MODEL)
        
        total_tokens += (cost_info["input_tokens"] + cost_info["output_tokens"])
        total_cost += cost_info["total_cost"]

    # Trả về thống kê
    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history
    }
    raise NotImplementedError("Implement run_assistant")


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        comparison = compare_models(prompt)
        results.append({**comparison, "prompt": prompt})
    return results
    raise NotImplementedError("Implement batch_compare")


def format_comparison_table(results: list[dict]) -> str:
    def shorten(text: str) -> str:
        return text if len(text) <= 40 else f"{text[:37]}..."

    rows = ["Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency"]
    for result in results:
        rows.append(
            " | ".join(
                [
                    shorten(result["prompt"]),
                    shorten(result["gpt4o_response"]),
                    shorten(result["mini_response"]),
                    f"{result['gpt4o_latency']:.2f}s",
                    f"{result['mini_latency']:.2f}s",
                ]
            )
        )
    return "\n".join(rows)
    raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
