import math
import csv

def load_csv(filename):
    dataset = []
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')             
            header = next(csv_reader)  

            for row in csv_reader:
                if row:  
                    dataset.append(row)

        print(f"Berhasil memuat data dari {filename} dengan {len(dataset)} baris.")
        return dataset, header
    except FileNotFoundError:
        print(f"Error: File {filename} Tidak Ditemukan.")
        return [], []
    

def train_categorical_nb(dataset):
    # --- 1. Memisahkan data berdasarkan label kelas ---
    total_rows = len(dataset)
    separated = {} 
    
    for row in dataset:
        label = row[-1]
        if label not in separated:
            separated[label] = []
        separated[label].append(row)

    # --- 2. Menghitung probabilitas prior  ---
    priors = {} 
    for label, rows in separated.items():
        priors[label] = len(rows) / total_rows 

    # --- 3. Menghitung likelihood ---   
    likelihoods = {}
    features_vocab = {}

    # Hitung jumlah fitur (Total kolom - 1 kolom label)
    num_features = len(dataset[0]) - 1

    # Cari vocabulary unik per kolom dulu
    for i in range(num_features):
        column_values = [row[i] for row in dataset] 
        features_vocab[i] = list(set(column_values))

    # Struktur Likelihood
    for label, rows in separated.items():
        likelihoods[label] = {}

        for i in range(num_features):
            likelihoods[label][i] = {}

            # Inisialisasi 0 untuk semua kemungkinan nilai
            for val in features_vocab[i]:
                likelihoods[label][i][val] = 0
            
            # Hitung frekuensi 
            for row in rows:
                feature_val = row[i]
                likelihoods[label][i][feature_val] += 1
                
    return priors, likelihoods, separated, features_vocab
        

def predict_row(input_row, header, model_data, row_number):
    priors, likelihoods, separated, features_vocab = model_data
    
    # --- MENAMPILKAN PERHITUNGAN (Syarat Poin 3) ---
    print(f"\n{'='*50}")
    print(f"DATA TESTING KE-{row_number}")
    print(f"Input Fitur: {input_row}")
    
    # Hitung Posterior
    best_label = None
    best_score = -float('inf') 
    
    alpha = 1
    num_features_train = len(features_vocab)
    
    # Cek setiap kelas
    print("-" * 50)
    for label in priors:
        total_class_count = len(separated[label])
        log_prob_sum = math.log(priors[label]) 
        
        calc_str = [] # Untuk menyimpan teks perhitungan agar bisa diprint
        
        for i in range(num_features_train):
            if i < len(input_row):
                feature_val = input_row[i]
                count = likelihoods[label][i].get(feature_val, 0)
                k = len(features_vocab[i])
                prob = (count + alpha) / (total_class_count + (alpha * k))
                log_prob_sum += math.log(prob)
                
                # Menampilkan Probabilitas per atribut (Poin 3)
                col_name = header[i] if header else f"F{i}"
                calc_str.append(f"P({col_name}={feature_val}|{label})={prob:.3f}")
        
        # Print perhitungan detail
        print(f"Kelas {label} -> {', '.join(calc_str)}")
        print(f"   >> Skor Akhir (Log): {log_prob_sum:.4f}")

        if log_prob_sum > best_score:
            best_score = log_prob_sum
            best_label = label
            
    return best_label

# ==============================================================================
# 4. EKSEKUSI UTAMA (DENGAN HITUNG AKURASI)
# ==============================================================================

# Ganti nama file sesuai punya kamu
file_training = 'Data_Training.csv' 
file_testing  = 'Data_Test.csv' 

# 1. Load Training
train_data, train_header = load_csv(file_training)

if train_data:
    # 2. Latih Model
    model = train_categorical_nb(train_data)
    
    # 3. Load Testing
    print(f"\n[INFO] Membaca data testing...")
    test_data, test_header = load_csv(file_testing)
    
    if test_data:
        jumlah_benar = 0
        total_data = len(test_data)
        
        print(f"\n[INFO] Mulai Pengujian pada {total_data} data...")
        
        # 4. Loop Testing
        for i, row in enumerate(test_data):
            # Asumsi: File Testing JUGA punya kolom Jawaban (Label) di akhir
            # Pisahkan Fitur dan Jawaban Asli
            input_fitur = row[:-1]  
            jawaban_asli = row[-1]  
            
            # Lakukan Prediksi
            prediksi_sistem = predict_row(input_fitur, train_header, model, i+1)
            
            # Cek Apakah Benar?
            status = "SALAH"
            if prediksi_sistem == jawaban_asli:
                jumlah_benar += 1
                status = "BENAR"
                
            print(f"\n>>> HASIL: Prediksi [{prediksi_sistem}] vs Asli [{jawaban_asli}] -> {status}")
            
        # 5. HITUNG AKURASI (Syarat Poin 5)
        akurasi = (jumlah_benar / total_data) * 100
        
        print(f"\n{'='*50}")
        print("LAPORAN AKURASI SISTEM")
        print(f"{'='*50}")
        print(f"Jumlah Data Uji   : {total_data}")
        print(f"Prediksi Benar    : {jumlah_benar}")
        print(f"Prediksi Salah    : {total_data - jumlah_benar}")
        print(f"-----------------------------")
        print(f"TOTAL AKURASI     : {akurasi:.2f}%")
        print(f"{'='*50}")
        
    else:
        print("File testing kosong.")
else:
    print("File training kosong.")