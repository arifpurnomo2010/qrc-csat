# Gunakan base image Python yang ringan
FROM python:3.9-slim

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Buat file .dockerignore untuk mencegah penyalinan yang tidak perlu
# dan menjaga image tetap kecil
COPY .dockerignore .dockerignore

# Salin semua file dari folder lokal ke direktori kerja /app di dalam container
# File yang tercantum di .dockerignore akan diabaikan
COPY . .

# Install dependensi Python dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Beri tahu Docker bahwa container akan listen di port 8000
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container dimulai
# Kita menggunakan Gunicorn, bukan 'python app.py'
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:server"]