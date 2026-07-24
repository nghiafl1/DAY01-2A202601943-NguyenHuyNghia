# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi Temperature = 0.0, câu trả lời rất máy móc, rập khuôn và luôn giống hệt nhau ở các lần gọi. Ở 0.5, văn phong tự nhiên hơn nhưng vẫn bám sát thực tế. Ở 1.0, AI sáng tạo, dùng từ ngữ phong phú hơn. Ở 1.5, văn bản bắt đầu trở nên lộn xộn, sai ngữ pháp hoặc sinh ra thông tin ảo (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature ở mức rất thấp (0.0 đến 0.2). Chatbot CSKH cần sự chính xác tuyệt đối, nhất quán trong chính sách và quy định, không được phép "sáng tạo" hay bịa ra thông tin ưu đãi/giá cả không có thật.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Dựa trên bảng giá, GPT-4o đắt hơn GPT-4o-mini khoảng 16 lần (output: $0.010/1k so với $0.0006/1k). Nên dùng GPT-4o: Các tác vụ yêu cầu suy luận logic phức tạp, viết code khó, hoặc phân tích báo cáo tài chính. Nên dùng GPT-4o-mini: Chatbot hỏi đáp thông thường, tóm tắt đoạn văn ngắn, trích xuất dữ liệu (NER) để tiết kiệm chi phí khi scale.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona "giáo viên", model dùng câu ngắn, từ vựng đời thường và ví dụ trực quan (như cuốn sổ cái dùng chung của lớp). Với persona "chuyên gia", model dùng từ khóa chuyên ngành (mã hóa, đồng thuận, phi tập trung, ledger) và câu văn dài, cấu trúc phức tạp. System prompt hoạt động như một "đạo diễn", thiết lập ranh giới từ vựng, độ sâu kỹ thuật và giọng điệu trước khi model xử lý câu hỏi của người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Phương pháp đếm từ (từ / 0.75) thường đánh giá thấp số token thực tế của tiếng Việt, độ chênh lệch có thể từ 50% - 100%. Lý do là bộ mã hóa (tokenizer) của các LLM được tối ưu cho tiếng Anh (thường 1 từ = 1-1.3 token). Tiếng Việt là ngôn ngữ đơn âm tiết có nhiều dấu câu, tokenizer thường phải chẻ một từ tiếng Việt (ví dụ "khuyển") thành 2-3 sub-word/token, dẫn đến tốn kém token hơn nhiều so với tiếng Anh.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming cực kỳ quan trọng cho các ứng dụng tương tác trực tiếp với người dùng (chatbot UI, trợ lý giọng nói) vì nó giảm thời gian chờ đợi nhận byte đầu tiên (TTFB) xuống mức 0, tránh cảm giác app bị treo. Ngược lại, non-streaming phù hợp hơn cho các tác vụ chạy ngầm (batch processing), trích xuất JSON, hoặc tóm tắt tài liệu tự động, nơi ta chỉ cần toàn bộ dữ liệu hoàn chỉnh để đẩy vào pipeline tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Nếu hàng nghìn client cùng dùng delay cố định (VD: 1 giây), khi API phục hồi, toàn bộ nghìn request đó sẽ cùng ập tới server ngay lập tức ở giây tiếp theo, gây ra hiện tượng "Thundering Herd" làm sập server lần nữa. Exponential backoff (cộng thêm chút random jitter) giúp phân tán thời điểm gửi lại request của các client, giảm áp lực đột ngột lên hệ thống API và tăng tỷ lệ thành công.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: "Bạn là trợ lý kỹ thuật phần mềm. Hãy trả lời ngắn gọn, luôn cung cấp code mẫu nếu có thể và giải thích bằng tiếng Việt bình dân.". Lựa chọn: "Ngắn gọn" giúp tiết kiệm output token (giảm chi phí và độ trễ). "Tiếng Việt bình dân" giúp tránh việc AI dịch word-by-word các thuật ngữ chuyên ngành một cách máy móc, giữ cho câu trả lời dễ đọc.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là bộ nhớ (history) chỉ giữ được 3 lượt gần nhất, nếu người dùng hỏi lại một ngữ cảnh ở đầu cuộc hội thoại, bot sẽ quên mất. Cải thiện: Thay vì cắt cứng history = history[-6:], có thể tích hợp một tác vụ chạy ngầm dùng model nhỏ (mini) để liên tục tóm tắt các cuộc hội thoại cũ thành một đoạn văn ngắn, sau đó nhồi đoạn tóm tắt này vào đầu system prompt để bot có "trí nhớ dài hạn" mà không bị phình token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
