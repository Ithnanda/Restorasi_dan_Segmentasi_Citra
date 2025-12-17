import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# FUNGSI DETEKSI TEPI
def roberts(img):
    gx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)
    
    gy = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)
    
    g1 = cv2.filter2D(img, cv2.CV_32F, gx)
    g2 = cv2.filter2D(img, cv2.CV_32F, gy)
    
    magnitude = np.sqrt(g1**2 + g2**2)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def prewitt(img):
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    
    ky = np.array([[-1, -1, -1],
                   [0,   0,  0],
                   [1,   1,  1]], dtype=np.float32)
    
    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def sobel(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

def frei_chen(img):
    a = 1 / np.sqrt(2)
    
    kx = np.array([[-1, 0, 1],
                   [-a, 0, a],
                   [-1, 0, 1]], dtype=np.float32)
    
    ky = np.array([[-1, -a, -1],
                   [0,   0,  0],
                   [1,   a,  1]], dtype=np.float32)
    
    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)

# FUNGSI PERHITUNGAN MSE
def hitung_mse(img_original, img_hasil):
    """
    Menghitung Mean Squared Error antara citra original dan hasil segmentasi
    """
    mse = np.mean((img_original.astype(float) - img_hasil.astype(float)) ** 2)
    return mse

# FUNGSI VISUALISASI DENGAN MSE
def visualisasi_hasil(nama_citra, img_original, hasil_dict, mse_dict):
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    metode_urutan = ['roberts', 'prewitt', 'sobel', 'frei_chen']
    for idx, metode in enumerate(metode_urutan, start=2):
        if metode in hasil_dict:
            plt.subplot(2, 3, idx)
            plt.imshow(hasil_dict[metode], cmap='gray')
            mse_value = mse_dict[metode]
            plt.title(f'{metode.capitalize()}\nMSE: {mse_value:.2f}', 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
    
    plt.suptitle(f'Hasil Segmentasi - {nama_citra}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    nama_file = f'visualisasi_{nama_citra}.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f"   Visualisasi disimpan: {nama_file}")
    plt.show()

def perbandingan_metode(nama_citra, img_original, hasil_dict, mse_dict):
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(img_original, cmap='gray')
    plt.title('Original', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    metode_urutan = ['roberts', 'prewitt', 'sobel', 'frei_chen']
    for idx, metode in enumerate(metode_urutan, start=2):
        if metode in hasil_dict:
            plt.subplot(1, 5, idx)
            plt.imshow(hasil_dict[metode], cmap='gray')
            mse_value = mse_dict[metode]
            plt.title(f'{metode.capitalize()}\nMSE: {mse_value:.2f}', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
    
    plt.suptitle(f'Perbandingan Metode Deteksi Tepi - {nama_citra}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    nama_file = f'perbandingan_{nama_citra}.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f"   Perbandingan disimpan: {nama_file}")
    plt.show()

# PROGRAM UTAMA
def main():
    os.makedirs("segmentasi_output", exist_ok=True)
    print("Folder 'segmentasi_output' siap digunakan\n")
    
    citra_list = {
        "potrait_salt_pepper": "salt_pepper_noise.jpg",
        "potrait_gaussian": "gaussian_noise.jpg",
        "landscape_salt_pepper": "landscape_salt_pepper.jpg",
        "landscape_gaussian": "landscape_gaussian.jpg",
        "grayscale_potrait": "grayscale_potrait.jpg",
        "grayscale_landscape": "grayscale_landscape.jpg",
        "potrait_gs_filter_mean": "grayscale_potrait_gaussian_mean_filter_best.jpg",
        "potrait_sp_filter_median": "grayscale_potrait_salt_&_pepper_median_filter_best.jpg",
        "landscape_gs_filter_median": "grayscale_landscape_gaussian_median_filter_best.jpg",
        "landscape_sp_filter_median": "grayscale_landscape_salt_&_pepper_median_filter_best.jpg"
    }
    
    metode_list = {
        "roberts": roberts,
        "prewitt": prewitt,
        "sobel": sobel,
        "frei_chen": frei_chen
    }
    
    print("=" * 70)
    print("SEGMENTASI CITRA - DETEKSI TEPI DENGAN MSE")
    print("=" * 70)
    print(f"\nJumlah citra: {len(citra_list)}")
    print(f"Jumlah metode: {len(metode_list)}")
    print(f"Total output: {len(citra_list) * len(metode_list)} file\n")
    
    print("Metode yang digunakan:")
    for nama in metode_list.keys():
        print(f"  - {nama.capitalize()}")
    
    print("\n" + "=" * 70)
    print("MEMULAI PROSES SEGMENTASI")
    print("=" * 70 + "\n")
    
    hasil_segmentasi = {}
    mse_results = {}
    daftar_file_output = []
    
    for nama_citra, path_citra in citra_list.items():
        img = cv2.imread(path_citra, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"File tidak ditemukan: {path_citra}\n")
            continue
        
        print(f"Memproses: {nama_citra} ({img.shape[1]}x{img.shape[0]})")
        hasil_segmentasi[nama_citra] = {}
        mse_results[nama_citra] = {}
        
        for nama_metode, fungsi_metode in metode_list.items():
            hasil = fungsi_metode(img)
            hasil_segmentasi[nama_citra][nama_metode] = hasil
            
            # Hitung MSE
            mse = hitung_mse(img, hasil)
            mse_results[nama_citra][nama_metode] = mse
            
            nama_file = f"{nama_citra}_{nama_metode}.jpg"
            path_output = os.path.join("segmentasi_output", nama_file)
            cv2.imwrite(path_output, hasil)
            
            print(f"   {nama_metode.capitalize()}: {nama_file} | MSE: {mse:.2f}")
            daftar_file_output.append(('segmentasi', nama_file))
        
        # Tampilkan metode terbaik (MSE terendah)
        metode_terbaik = min(mse_results[nama_citra], key=mse_results[nama_citra].get)
        mse_terbaik = mse_results[nama_citra][metode_terbaik]
        print(f"   >>> Metode terbaik: {metode_terbaik.capitalize()} (MSE: {mse_terbaik:.2f})")
        
        visualisasi_hasil(nama_citra, img, hasil_segmentasi[nama_citra], 
                         mse_results[nama_citra])
        perbandingan_metode(nama_citra, img, hasil_segmentasi[nama_citra], 
                           mse_results[nama_citra])
        
        daftar_file_output.append(('visualisasi', f'visualisasi_{nama_citra}.png'))
        daftar_file_output.append(('visualisasi', f'perbandingan_{nama_citra}.png'))
        
        print()
    
    print("=" * 70)
    print("RINGKASAN MSE UNTUK SEMUA CITRA")
    print("=" * 70)
    for nama_citra in mse_results:
        print(f"\n{nama_citra}:")
        for metode in ['roberts', 'prewitt', 'sobel', 'frei_chen']:
            if metode in mse_results[nama_citra]:
                mse = mse_results[nama_citra][metode]
                print(f"  {metode.capitalize():15s}: {mse:10.2f}")
    
    print("\n" + "=" * 70)
    print("DAFTAR FILE OUTPUT")
    print("=" * 70)
    
    print("\nHasil Segmentasi (folder: segmentasi_output/):")
    for tipe, nama_file in daftar_file_output:
        if tipe == 'segmentasi':
            print(f"  - {nama_file}")
    
    print("\nVisualisasi (folder: utama):")
    for tipe, nama_file in daftar_file_output:
        if tipe == 'visualisasi':
            print(f"  - {nama_file}")
    
    print(f"\nTotal file yang dihasilkan: {len(daftar_file_output)}\n")

if __name__ == "__main__":
    main()