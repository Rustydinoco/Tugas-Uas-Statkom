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
        

def predict_detailed(dataset, input_row, header):
    priors, likelihoods, separated, features_vocab = train_categorical_nb(dataset)
    
    print(f"\n{'='*40}")
    print(f"INPUT DATA: {input_row}")
    print(f"{'='*40}\n")

    # [Bagian A]
    print("a. Prior Probabilities P(Class):")
    for label, prob in priors.items(): 
        print(f"   P({label}) = {prob:.4f}") 
    print("-" * 20)

    # [Bagian B]
    print("b. Conditional Probabilities P(Fitur|Class):")
    alpha = 1 
    
    num_features = len(features_vocab)
    
    class_scores = {}
    
    for label in priors:
        total_class_count = len(separated[label])
        log_prob_sum = math.log(priors[label]) 
        
        calc_details = []
        
        for i in range(num_features):
            # Cek apakah input row cukup panjang
            if i < len(input_row):
                feature_val = input_row[i]
                feature_name = header[i] if header else f"Fitur {i}"
                
                # Ambil count, default 0 jika nilai tidak ada di training
                count = likelihoods[label][i].get(feature_val, 0)
                
                k = len(features_vocab[i])
                prob = (count + alpha) / (total_class_count + (alpha * k))
                
                log_prob_sum += math.log(prob)
                
                calc_details.append(f"P({feature_name}={feature_val}|{label})={prob:.3f}")
        
        print(f"   Class {label}: \n      {', '.join(calc_details)}")
        class_scores[label] = log_prob_sum 

    print("-" * 20)

    # [Bagian C]
    print("c. Posterior Probabilities (Log Scale):")
    best_label = None
    best_score = -float('inf') 
    
    for label, score in class_scores.items():
        print(f"   Posterior({label}) = {score:.4f}")
        if score > best_score:
            best_score = score
            best_label = label
            
    print("-" * 20)

    # [Bagian D]
    print("d. Hasil Segmentasi (Prediksi):")
    print(f"   Data masuk ke dalam Segmentasi: [{best_label}]")
    print(f"{'='*40}")


# --- EKSEKUSI UTAMA ---
nama_file = 'Data Training - Sheet1.csv' 
dataset_excel, header = load_csv(nama_file)

if dataset_excel:

    raw_input = ['Male', 'Yes', '18', 'Yes', 'Doctor', '8.0', 'Average', '3.0', 'Cat_2','B'] 
    
    input_user = raw_input[:-1] 
    
    predict_detailed(dataset_excel, input_user, header)