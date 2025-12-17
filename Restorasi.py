import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import minimum_filter, maximum_filter

# FUNGSI PENAMBAHAN NOISE
def tambah_salt_pepper(img, salt_prob=0.02, pepper_prob=0.02):
    """Menambahkan Salt & Pepper noise"""
    noisy = img.copy()
    
    # Salt noise (white)
    salt = np.random.random(img.shape) < salt_prob
    noisy[salt] = 255
    
    # Pepper noise (black)
    pepper = np.random.random(img.shape) < pepper_prob
    noisy[pepper] = 0
    
    return noisy

def tambah_gaussian(img, mean=0, sigma=25):
    """Menambahkan Gaussian noise"""
    gaussian = np.random.normal(mean, sigma, img.shape)
    noisy = img + gaussian
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# FUNGSI RESTORASI CITRA
def mean_filter(img, kernel_size=3):
    """Filter Mean - mengurangi noise dengan rata-rata"""
    return cv2.blur(img, (kernel_size, kernel_size))

def median_filter(img, kernel_size=3):
    """Filter Median - efektif untuk salt & pepper noise"""
    return cv2.medianBlur(img, kernel_size)

def min_filter(img, kernel_size=3):
    """Filter Min - mengambil nilai minimum dalam window"""
    return minimum_filter(img, size=kernel_size)

def max_filter(img, kernel_size=3):
    """Filter Max - mengambil nilai maksimum dalam window"""
    return maximum_filter(img, size=kernel_size)

# FUNGSI PERHITUNGAN METRIK
def hitung_mse(img_original, img_restored):
    """Mean Squared Error - semakin rendah semakin baik"""
    mse = np.mean((img_original.astype(float) - img_restored.astype(float)) ** 2)
    return mse

# FUNGSI VISUALISASI
def visualisasi_hasil(nama_citra, img_original, hasil_dict, mse_dict, noise_type, level):
    """Visualisasi hasil restorasi dengan MSE seperti segmentasi"""
    is_grayscale = len(img_original.shape) == 2
    
    plt.figure(figsize=(16, 8))
    
    # Original
    plt.subplot(2, 3, 1)
    if is_grayscale:
        plt.imshow(img_original, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Hasil restorasi
    metode_urutan = ['Mean Filter', 'Median Filter', 'Min Filter', 'Max Filter']
    for idx, metode in enumerate(metode_urutan, start=2):
        if metode in hasil_dict:
            plt.subplot(2, 3, idx)
            if is_grayscale:
                plt.imshow(hasil_dict[metode], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(hasil_dict[metode], cv2.COLOR_BGR2RGB))
            mse_value = mse_dict[metode]
            plt.title(f'{metode}\nMSE: {mse_value:.2f}', 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
    
    plt.suptitle(f'Hasil Restorasi - {nama_citra} ({noise_type} Level {level})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    nama_file = f'visualisasi_{nama_citra}_{noise_type.lower().replace(" ", "_")}_level{level}.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f"   Visualisasi disimpan: {nama_file}")
    plt.show()

def perbandingan_metode(nama_citra, img_original, hasil_dict, mse_dict, noise_type, level):
    """Perbandingan horizontal semua metode"""
    is_grayscale = len(img_original.shape) == 2
    
    plt.figure(figsize=(18, 4))
    
    # Original
    plt.subplot(1, 5, 1)
    if is_grayscale:
        plt.imshow(img_original, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Original', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Hasil restorasi
    metode_urutan = ['Mean Filter', 'Median Filter', 'Min Filter', 'Max Filter']
    for idx, metode in enumerate(metode_urutan, start=2):
        if metode in hasil_dict:
            plt.subplot(1, 5, idx)
            if is_grayscale:
                plt.imshow(hasil_dict[metode], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(hasil_dict[metode], cv2.COLOR_BGR2RGB))
            mse_value = mse_dict[metode]
            plt.title(f'{metode}\nMSE: {mse_value:.2f}', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
    
    plt.suptitle(f'Perbandingan Metode Restorasi - {nama_citra} ({noise_type} Level {level})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    nama_file = f'perbandingan_{nama_citra}_{noise_type.lower().replace(" ", "_")}_level{level}.png'
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    print(f"   Perbandingan disimpan: {nama_file}")
    plt.show()

# FUNGSI SIMPAN HASIL
def simpan_hasil_terbaik(nama_citra, img_terbaik, metode_terbaik, noise_type, level):
    """Simpan hanya citra hasil restorasi terbaik"""
    nama_file = f'{nama_citra}_{noise_type.lower().replace(" ", "_")}_level{level}_{metode_terbaik.lower().replace(" ", "_")}_best.jpg'
    cv2.imwrite(nama_file, img_terbaik)
    print(f"   Hasil terbaik disimpan: {nama_file}")

# PROGRAM UTAMA
def main():
    os.makedirs("restorasi_output", exist_ok=True)
    print("Folder 'restorasi_output' siap digunakan\n")
    
    # Daftar citra yang akan direstorasi
    citra_list = {
        # RGB
        "rgb_landscape": "landscape.jpg",  
        "rgb_potrait": "potrait.jpg",    
        # Grayscale
        "grayscale_landscape": "grayscale_landscape.jpg",
        "grayscale_potrait": "grayscale_potrait.jpg"
    }
    
    # Jenis noise dengan 2 level
    noise_configs = {
        "Salt & Pepper": {
            "level1": lambda img: tambah_salt_pepper(img, 0.01, 0.01),  # Level 1: rendah
            "level2": lambda img: tambah_salt_pepper(img, 0.05, 0.05)   # Level 2: tinggi
        },
        "Gaussian": {
            "level1": lambda img: tambah_gaussian(img, 0, 15),   # Level 1: rendah
            "level2": lambda img: tambah_gaussian(img, 0, 35)    # Level 2: tinggi
        }
    }
    
    # Daftar metode restorasi
    metode_list = {
        "Mean Filter": lambda img: mean_filter(img, 5),
        "Median Filter": lambda img: median_filter(img, 5),
        "Min Filter": lambda img: min_filter(img, 5),
        "Max Filter": lambda img: max_filter(img, 5)
    }
    
    print("=" * 70)
    print("RESTORASI CITRA RGB & GRAYSCALE DENGAN MSE")
    print("=" * 70)
    print(f"\nJumlah citra: {len(citra_list)}")
    print(f"  - RGB: 2 (landscape, potrait)")
    print(f"  - Grayscale: 2 (landscape, potrait)")
    print(f"\nJenis noise: 2 (Salt & Pepper, Gaussian)")
    print(f"Level noise: 2 (Level 1, Level 2)")
    print(f"Jumlah metode restorasi: {len(metode_list)}")
    print(f"\nTotal kombinasi restorasi: {len(citra_list)} citra × 2 noise × 2 level = 8 proses")
    
    print("\nMetode restorasi yang digunakan:")
    for nama in metode_list.keys():
        print(f"  - {nama}")
    
    print("\n" + "=" * 70)
    print("MEMULAI PROSES RESTORASI")
    print("=" * 70 + "\n")
    
    ringkasan_hasil = {}
    total_proses = 0
    
    for nama_citra, path_citra in citra_list.items():
        # Tentukan apakah RGB atau Grayscale
        is_grayscale = "grayscale" in nama_citra
        
        if is_grayscale:
            img_original = cv2.imread(path_citra, cv2.IMREAD_GRAYSCALE)
        else:
            img_original = cv2.imread(path_citra)
        
        if img_original is None:
            print(f"❌ File tidak ditemukan: {path_citra}\n")
            continue
        
        if is_grayscale:
            print(f"Memproses: {nama_citra} ({img_original.shape[1]}x{img_original.shape[0]}) - GRAYSCALE")
        else:
            print(f"Memproses: {nama_citra} ({img_original.shape[1]}x{img_original.shape[0]}x{img_original.shape[2]}) - RGB")
        
        ringkasan_hasil[nama_citra] = {}
        
        for noise_type, level_funcs in noise_configs.items():
            print(f"\n  Noise: {noise_type}")
            ringkasan_hasil[nama_citra][noise_type] = {}
            
            for level_name, noise_func in level_funcs.items():
                level_num = level_name.replace("level", "")
                print(f"\n    Level {level_num}:")
                
                # Tambahkan noise
                img_noisy = noise_func(img_original)
                
                # Simpan citra noisy di folder utama sebagai sampel
                nama_file_noisy_sampel = f"{nama_citra}_{noise_type.lower().replace(' ', '_')}_level{level_num}_noisy.jpg"
                cv2.imwrite(nama_file_noisy_sampel, img_noisy)
                print(f"      Sampel citra noisy disimpan: {nama_file_noisy_sampel}")
                
                # Juga simpan di folder restorasi_output
                nama_file_noisy = f"{nama_citra}_{noise_type.lower().replace(' ', '_')}_{level_name}_noisy.jpg"
                path_noisy = os.path.join("restorasi_output", nama_file_noisy)
                cv2.imwrite(path_noisy, img_noisy)
                
                hasil_restorasi = {}
                metrik_results = {}
                
                for nama_metode, fungsi_metode in metode_list.items():
                    # Proses restorasi
                    hasil = fungsi_metode(img_noisy)
                    hasil_restorasi[nama_metode] = hasil
                    
                    # Hitung metrik (dibandingkan dengan citra original)
                    mse = hitung_mse(img_original, hasil)
                    
                    metrik_results[nama_metode] = mse
                    
                    # Simpan hasil di folder restorasi_output
                    nama_file = f"{nama_citra}_{noise_type.lower().replace(' ', '_')}_{level_name}_{nama_metode.lower().replace(' ', '_')}.jpg"
                    path_output = os.path.join("restorasi_output", nama_file)
                    cv2.imwrite(path_output, hasil)
                    
                    print(f"      {nama_metode}: MSE: {mse:.2f}")
                
                # Simpan ringkasan
                ringkasan_hasil[nama_citra][noise_type][level_name] = metrik_results
                
                # Tampilkan metode terbaik (MSE terendah)
                metode_terbaik = min(metrik_results, key=metrik_results.get)
                mse_terbaik = metrik_results[metode_terbaik]
                print(f"      >>> Metode terbaik: {metode_terbaik} (MSE: {mse_terbaik:.2f})")
                
                # Simpan hanya hasil terbaik di folder utama
                img_terbaik = hasil_restorasi[metode_terbaik]
                simpan_hasil_terbaik(nama_citra, img_terbaik, metode_terbaik, noise_type, level_num)
                
                # Visualisasi dengan MSE
                visualisasi_hasil(nama_citra, img_original, hasil_restorasi, 
                                metrik_results, noise_type, level_num)
                perbandingan_metode(nama_citra, img_original, hasil_restorasi, 
                                  metrik_results, noise_type, level_num)
                
                total_proses += 1
        
        print()
    
    print("=" * 70)
    print("RINGKASAN MSE UNTUK SEMUA CITRA")
    print("=" * 70)
    
    for nama_citra in ringkasan_hasil:
        print(f"\n{nama_citra}:")
        for noise_type in ringkasan_hasil[nama_citra]:
            print(f"  {noise_type}:")
            for level_name in ringkasan_hasil[nama_citra][noise_type]:
                level_num = level_name.replace("level", "")
                print(f"    Level {level_num}:")
                for metode in ['Mean Filter', 'Median Filter', 'Min Filter', 'Max Filter']:
                    if metode in ringkasan_hasil[nama_citra][noise_type][level_name]:
                        mse = ringkasan_hasil[nama_citra][noise_type][level_name][metode]
                        print(f"      {metode:15s}: {mse:10.2f}")
    
    print("\n" + "=" * 70)
    print("DAFTAR FILE OUTPUT")
    print("=" * 70)
    
    print(f"\nTotal proses restorasi: {total_proses}")
    print(f"\nSampel Citra Noisy (folder utama): {total_proses} file")
    print(f"Hasil Restorasi Terbaik (folder utama): {total_proses} file")
    print(f"Visualisasi Grid 2x3 (folder utama): {total_proses} file")
    print(f"Perbandingan Horizontal (folder utama): {total_proses} file")
    print(f"\nSemua hasil restorasi (folder restorasi_output/): {total_proses * 4} file")
    print(f"Citra noisy backup (folder restorasi_output/): {total_proses} file")
    
    print("\n" + "=" * 70)
    print("PROSES SELESAI")
    print("=" * 70)

if __name__ == "__main__":

    main()
