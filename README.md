**DỰ ÁN MINH HỌA NHẬN DIỆN BÀN TAY THỜI GIAN THỰC**  
SỬ DỤNG **:contentReference[oaicite:0]{index=0} HANDS** KẾT HỢP VỚI **:contentReference[oaicite:1]{index=1}**,  
NGUỒN DỮ LIỆU ĐẦU VÀO LÀ **WEBCAM (HỖ TRỢ WEBCAM NGOÀI)**.

Trong quá trình thử nghiệm, việc chuyển ảnh sang grayscale làm giảm đáng kể độ chính xác nhận diện. Nguyên nhân là do MediaPipe Hands yêu cầu ảnh đầu vào ở không gian màu RGB, vì mô hình được huấn luyện dựa trên thông tin màu. Khi loại bỏ thông tin này, mô hình không còn đủ đặc trưng để phát hiện bàn tay một cách ổn định.
