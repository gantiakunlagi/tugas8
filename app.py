from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np
from PIL import Image
import os
import uuid
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'static/uploads'

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_tenun.pkl')
bundle = joblib.load(MODEL_PATH)
rf_model  = bundle['model']
scaler    = bundle['scaler']
CLASSES   = bundle['classes']
MODEL_ACC = bundle['accuracy']

MOTIF_INFO = {
    'Aceh_Pintu_Aceh': {
        'origin': 'Aceh', 'icon': '🟢', 'color': '#1A6B3A',
        'pattern': 'Motif pintu gerbang dengan ornamen bunga dan sulur',
        'desc': 'Batik khas Aceh dengan motif Pintu Aceh yang terinspirasi dari gapura tradisional. Kaya akan ornamen floral dan geometris.',
    },
    'Bali_Barong': {
        'origin': 'Bali', 'icon': '🟠', 'color': '#C85A00',
        'pattern': 'Figur Barong dengan detail wajah dan mahkota',
        'desc': 'Motif batik Bali yang menggambarkan Barong, makhluk mitologi pelindung dalam kepercayaan Hindu Bali.',
    },
    'Bali_Merak': {
        'origin': 'Bali', 'icon': '🔵', 'color': '#1A5A8B',
        'pattern': 'Burung merak dengan ekor mekar dan bulu berwarna-warni',
        'desc': 'Batik Bali bermotif burung merak yang melambangkan keindahan dan kemakmuran. Warnanya cerah dan penuh detail.',
    },
    'DKI_Ondel_Ondel': {
        'origin': 'DKI Jakarta (Betawi)', 'icon': '🔴', 'color': '#C41E3A',
        'pattern': 'Figur Ondel-Ondel dengan pakaian adat Betawi',
        'desc': 'Batik Betawi dengan ikon Ondel-Ondel, boneka raksasa khas Jakarta yang digunakan dalam perayaan adat.',
    },
    'Jawa_Barat_Megamendung': {
        'origin': 'Cirebon, Jawa Barat', 'icon': '🟦', 'color': '#1A3A8B',
        'pattern': 'Motif awan bertingkat berbentuk seperti megamendung',
        'desc': 'Salah satu motif batik paling ikonik dari Cirebon. Pola awan berlapis melambangkan hujan pembawa kesuburan.',
    },
    'Jawa_Timur_Pring': {
        'origin': 'Jawa Timur', 'icon': '🟩', 'color': '#2D6B1A',
        'pattern': 'Motif bambu (pring) dengan daun dan ruas bambu',
        'desc': 'Batik khas Jawa Timur bermotif bambu (pring dalam bahasa Jawa), melambangkan keteguhan dan kelenturan.',
    },
    'Kalimantan_Dayak': {
        'origin': 'Kalimantan', 'icon': '🟤', 'color': '#6B3A1A',
        'pattern': 'Motif tribal Dayak dengan ornamen geometris dan figur hewan',
        'desc': 'Batik terinspirasi dari seni tenun dan ukiran suku Dayak Kalimantan dengan corak tribal yang kuat.',
    },
    'Lampung_Gajah': {
        'origin': 'Lampung', 'icon': '🐘', 'color': '#5C5C1A',
        'pattern': 'Figur gajah Sumatera dengan ornamen khas Lampung',
        'desc': 'Batik Lampung dengan motif gajah Sumatera yang menjadi simbol kebesaran dan kekuatan daerah.',
    },
    'Madura_Mataketeran': {
        'origin': 'Madura, Jawa Timur', 'icon': '🟥', 'color': '#8B1A1A',
        'pattern': 'Motif mata keteran dengan warna merah, hitam, kuning yang mencolok',
        'desc': 'Batik Madura terkenal dengan warna cerah dan kontras tinggi. Motif Mataketeran memiliki corak geometris yang kuat.',
    },
    'Maluku_Pala': {
        'origin': 'Maluku', 'icon': '🍃', 'color': '#1A6B5A',
        'pattern': 'Motif buah dan daun pala dengan sulur melingkar',
        'desc': 'Batik Maluku terinspirasi dari tanaman pala, rempah ikonik Kepulauan Maluku yang bernilai sejarah tinggi.',
    },
    'NTB_Lumbung': {
        'origin': 'Nusa Tenggara Barat', 'icon': '🏠', 'color': '#8B6B1A',
        'pattern': 'Motif lumbung (tempat penyimpanan padi) khas Sasak',
        'desc': 'Batik NTB dengan motif lumbung, bangunan tradisional Lombok yang melambangkan kemakmuran dan ketahanan pangan.',
    },
    'Papua_Asmat': {
        'origin': 'Papua', 'icon': '🎭', 'color': '#3A1A0A',
        'pattern': 'Ukiran Asmat dengan figur manusia dan motif tribal',
        'desc': 'Terinspirasi dari seni ukir Suku Asmat Papua yang terkenal di dunia. Motif penuh dengan simbol kehidupan dan leluhur.',
    },
    'Papua_Cendrawasih': {
        'origin': 'Papua', 'icon': '🦜', 'color': '#6B1A6B',
        'pattern': 'Burung cendrawasih dengan bulu ekor panjang berwarna-warni',
        'desc': 'Batik Papua bermotif burung Cendrawasih, "Bird of Paradise" yang menjadi simbol keindahan alam Papua.',
    },
    'Papua_Tifa': {
        'origin': 'Papua', 'icon': '🥁', 'color': '#6B3A00',
        'pattern': 'Alat musik Tifa dengan ornamen geometris Papua',
        'desc': 'Batik Papua dengan motif alat musik Tifa, instrumen perkusi tradisional yang digunakan dalam upacara adat.',
    },
    'Solo_Parang': {
        'origin': 'Solo (Surakarta), Jawa Tengah', 'icon': '⚔️', 'color': '#1A1A1A',
        'pattern': 'Garis diagonal berulang menyerupai ombak atau bilah parang',
        'desc': 'Salah satu motif batik tertua dan paling sakral dari Solo. Motif Parang melambangkan semangat dan keteguhan jiwa.',
    },
    'Sulawesi_Selatan_Lontara': {
        'origin': 'Sulawesi Selatan (Bugis-Makassar)', 'icon': '📜', 'color': '#1A4A6B',
        'pattern': 'Aksara Lontara dan ornamen khas Bugis-Makassar',
        'desc': 'Batik terinspirasi dari aksara Lontara, tulisan tradisional suku Bugis dan Makassar Sulawesi Selatan.',
    },
    'Sumatera_Barat_Rumah_Minang': {
        'origin': 'Sumatera Barat (Minangkabau)', 'icon': '🏛️', 'color': '#8B1A6B',
        'pattern': 'Siluet Rumah Gadang dengan atap bergonjong khas Minang',
        'desc': 'Batik Minangkabau dengan motif Rumah Gadang, rumah adat dengan atap bergonjong yang ikonik.',
    },
    'Sumatera_Utara_Boraspati': {
        'origin': 'Sumatera Utara (Batak)', 'icon': '🦎', 'color': '#6B1A1A',
        'pattern': 'Motif cicak Boraspati dengan ornamen Batak',
        'desc': 'Batik Batak dengan motif Boraspati (cicak), hewan yang dianggap suci dan membawa keberuntungan dalam adat Batak.',
    },
    'Yogyakarta_Kawung': {
        'origin': 'Yogyakarta', 'icon': '⭕', 'color': '#1A1A6B',
        'pattern': 'Lingkaran oval berempat menyerupai buah kolang-kaling',
        'desc': 'Motif batik klasik Yogyakarta yang terinspirasi dari buah aren (kawung). Salah satu motif tertua dalam tradisi batik Jawa.',
    },
    'Yogyakarta_Parang': {
        'origin': 'Yogyakarta', 'icon': '〰️', 'color': '#2D2D2D',
        'pattern': 'Diagonal zigzag menyerupai gelombang parang rusak',
        'desc': 'Motif Parang khas Yogyakarta yang dulunya hanya boleh dikenakan oleh keluarga keraton. Melambangkan kekuasaan dan kewibawaan.',
    },
}

def extract_features(img, size=64):
    img = img.resize((size, size)).convert('RGB')
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
        features += list(hist / hist.sum())
    return np.array(features)

@app.route('/')
def index():
    samples = []
    sample_dir = os.path.join(app.static_folder, 'samples')
    if os.path.exists(sample_dir):
        for cls in CLASSES:
            for ext in ['jpg', 'jpeg', 'png']:
                fname = f'{cls.lower()}.{ext}'
                if os.path.exists(os.path.join(sample_dir, fname)):
                    samples.append({'name': cls, 'file': fname})
                    break
    return render_template('index.html',
                           classes=CLASSES,
                           model_acc=round(MODEL_ACC * 100, 2),
                           samples=samples,
                           motif_info=MOTIF_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    allowed = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Format file tidak didukung: .{ext}'}), 400
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        fname = f'{uuid.uuid4().hex}.{ext}'
        save_path = os.path.join(app.static_folder, 'uploads', fname)
        img.save(save_path)
        feats = extract_features(img).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        proba = rf_model.predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_class = CLASSES[pred_idx]
        confidence = float(proba[pred_idx])
        class_probs = [
            {'name': CLASSES[i], 'prob': round(float(proba[i]) * 100, 1)}
            for i in range(len(CLASSES))
        ]
        class_probs.sort(key=lambda x: x['prob'], reverse=True)
        default_info = {'origin': 'Nusantara', 'icon': '🧵', 'color': '#8B7355',
                        'pattern': '-', 'desc': f'Motif batik {pred_class}.'}
        info = MOTIF_INFO.get(pred_class, default_info)
        return jsonify({
            'success': True,
            'prediction': pred_class,
            'confidence': round(confidence * 100, 2),
            'image_url': url_for('static', filename=f'uploads/{fname}'),
            'class_probs': class_probs,
            'info': info,
            'model_accuracy': round(MODEL_ACC * 100, 2),
        })
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/about')
def about():
    return render_template('about.html', classes=CLASSES,
                           motif_info=MOTIF_INFO,
                           model_acc=round(MODEL_ACC * 100, 2))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)