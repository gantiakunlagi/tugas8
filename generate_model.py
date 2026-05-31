"""
Train Random Forest dari dataset raw_batik_v2 yang asli.
Struktur folder:
  dataset/raw_batik_v2/train/<Kelas>/*.jpg
  dataset/raw_batik_v2/test/<Kelas>/*.jpg
"""
import os
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

DATASET_DIR  = 'dataset/raw_batik_v2'
TRAIN_DIR    = os.path.join(DATASET_DIR, 'train')
TEST_DIR     = os.path.join(DATASET_DIR, 'test')
MODEL_OUTPUT = 'model/rf_tenun.pkl'
SAMPLE_DIR   = 'static/samples'
IMG_SIZE     = 64

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

def extract_features(img_path, size=IMG_SIZE):
    try:
        img = Image.open(img_path).resize((size, size)).convert('RGB')
    except Exception as e:
        print(f"  [SKIP] {img_path}: {e}")
        return None
    arr = np.array(img, dtype=float)
    features = []
    for c in range(3):
        ch = arr[:, :, c]
        features += [ch.mean(), ch.std(), ch.min(), ch.max(),
                     np.percentile(ch, 25), np.percentile(ch, 75)]
    gray = arr.mean(axis=2)
    features += [np.mean(np.abs(np.diff(gray, axis=0))),
                 np.mean(np.abs(np.diff(gray, axis=1)))]
    block = 16
    for bi in range(0, size, block):
        for bj in range(0, size, block):
            patch = gray[bi:bi+block, bj:bj+block]
            features += [patch.mean(), patch.std()]
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 255))
        features += list(hist / (hist.sum() + 1e-9))
    return np.array(features)

def load_split(split_dir, classes):
    X, y = [], []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
        print(f"  {cls}: {len(files)} gambar")
        for fname in files:
            feat = extract_features(os.path.join(cls_dir, fname))
            if feat is not None:
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)

CLASSES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"\nDitemukan {len(CLASSES)} kelas: {', '.join(CLASSES)}\n")

print("Memuat data TRAIN...")
X_train, y_train = load_split(TRAIN_DIR, CLASSES)

print("\nMemuat data TEST...")
X_test, y_test = load_split(TEST_DIR, CLASSES)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]} | Fitur: {X_train.shape[1]}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\nMelatih Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Test Set: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=CLASSES))

joblib.dump({'model': rf, 'scaler': scaler, 'classes': CLASSES, 'accuracy': acc}, MODEL_OUTPUT)
print(f"Model disimpan ke: {MODEL_OUTPUT}")

print("\nMenyimpan contoh gambar...")
for cls in CLASSES:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if files:
        try:
            img = Image.open(os.path.join(cls_dir, files[0])).resize((256,256)).convert('RGB')
            img.save(os.path.join(SAMPLE_DIR, f"{cls.lower()}.jpg"))
        except: pass

print("Selesai! Jalankan: python app.py")
