import cv2 as cv
import numpy as np
from PIL import Image

def apply_conv(img, conv_type):
  kernels = {
    # Kernel "Original" hanya sebagai placeholder, tidak melakukan perubahan (akan digunakan gambar asli).
    'Original': np.array([[0]]),

    # Kernel "Blur (Mean)" untuk meratakan piksel dengan mengambil rata-rata dari 9 piksel sekitarnya (3x3).
    # Digunakan untuk menghaluskan atau mengurangi noise pada citra.
    'Blur (Mean)': np.ones((3, 3), np.float32) / 9,

    # Kernel "Gaussian Blur" menggunakan distribusi Gaussian untuk blur yang lebih halus dan alami.
    # Dibuat dengan mengalikan kernel Gaussian 1D secara baris dan kolom untuk mendapatkan 2D kernel.
    'Gaussian Blur': cv.getGaussianKernel(3, 0) @ cv.getGaussianKernel(3, 0).T,

    # Kernel "Sharpen" meningkatkan kontras dan menonjolkan tepi objek.
    # Piksel tengah bernilai tinggi (5), piksel sekitarnya negatif (-1), hasilnya tampak lebih tajam.
    'Sharpen': np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]]),

    # Kernel "Edge Detection (Sobel X)" mendeteksi tepi secara horizontal (vertikal pada gambar).
    # Nilai negatif di kiri dan positif di kanan akan memperlihatkan perbedaan intensitas horizontal.
    'Edge Detection (Sobel X)': np.array([[-1, 0, 1],
                                          [-2, 0, 2],
                                          [-1, 0, 1]]),

    # Kernel "Emboss" memberikan efek timbul atau 3D pada gambar.
    # Menghasilkan bayangan seolah-olah cahaya datang dari arah tertentu.
    'Emboss': np.array([[-2, -1, 0],
                        [-1, 1, 1],
                        [0, 1, 2]])
  }
  if conv_type not in kernels:
        return img

  if conv_type == 'Original':
      return img
  else:
      kernel = kernels[conv_type]
      return cv.filter2D(img, -1, kernel)
  



def image_procesing_pipeline(img, config={}, conv_type='sharpen', noise=False):
#   image = cv.imread(path)

  # reize agar ukran gambar sama
  image = cv.resize(img, (350, 350))

  # convert ke grayscale
  B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
  gray_image = (0.2126 * R + 0.7152 * G + 0.0722 * B).astype(np.uint8)

  # apply konvolusi (sharpen untuk defaultnya)
  image = np.array(apply_conv(gray_image, conv_type))

  # tambah noise
  if noise:
    row, col = image.shape
    amount = config.get('amount', 0.02)
    salt_vs_pepper = config.get('salt_vs_pepper', 0.05)

    # Salin gambar asli agar tidak merusak data asli
    noisy = np.copy(image)

    # Hitung jumlah pixel yang akan diubah menjadi "salt"
    num_salt = np.ceil(amount * image.size * salt_vs_pepper)

    # Tentukan koordinat acak untuk noise salt pada setiap dimensi gambar
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]

    # Set pixel di koordinat tersebut menjadi 255 putih
    noisy[coords[0], coords[1]] = 255

    # Hitung jumlah pixel yang akan diubah menjadi "pepper"
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_vs_pepper))

    # Tentukan koordinat acak untuk noise pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]

    # Set pixel di koordinat tersebut menjadi 0 (hitam)
    noisy[coords[0], coords[1]] = 0

    # Kembalikan gambar yang sudah diberi noise salt and pepper
    image = noisy

  return image