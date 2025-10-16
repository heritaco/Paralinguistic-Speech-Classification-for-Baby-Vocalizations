import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def extract_mfcc_features(self, y):
        """Extrae características MFCC"""
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        
        features = {}
        for i in range(20):
            features[f'mfcc_{i}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i}_std'] = mfcc_std[i]
            features[f'mfcc_{i}_delta_mean'] = mfcc_delta_mean[i]
        
        return features
    
    def extract_pitch_features(self, y):
        """Extrae características de pitch (f0)"""
        features = {}
        
        # Pitch usando piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sample_rate)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_median'] = np.median(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_median'] = 0
        
        return features
    
    def extract_energy_features(self, y):
        """Extrae características de energía"""
        features = {}
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def extract_spectral_features(self, y):
        """Extrae características espectrales"""
        features = {}
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        
        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        return features
    
    def extract_chroma_features(self, y):
        """Extrae características de chroma"""
        features = {}
        
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
        
        return features
    
    def extract_temporal_features(self, y):
        """Extrae características temporales"""
        features = {}
        
        # Duración
        features['duration'] = len(y) / self.sample_rate
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
        features['tempo'] = tempo[0] if len(tempo) > 0 else 0
        
        return features
    
    def extract_mel_spectrogram_features(self, y):
        """Extrae estadísticas del mel-espectrograma"""
        features = {}
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        features['mel_spec_mean'] = np.mean(mel_spec_db)
        features['mel_spec_std'] = np.std(mel_spec_db)
        features['mel_spec_max'] = np.max(mel_spec_db)
        features['mel_spec_min'] = np.min(mel_spec_db)
        
        return features
    
    def extract_all_features(self, audio_path):
        """Extrae todas las características de un archivo de audio"""
        try:
            # Cargar audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Normalizar audio
            y = librosa.util.normalize(y)
            
            # Extraer todas las características
            features = {}
            features['nombre_archivo'] = os.path.basename(audio_path)
            
            # MFCC
            features.update(self.extract_mfcc_features(y))
            
            # Pitch
            features.update(self.extract_pitch_features(y))
            
            # Energía
            features.update(self.extract_energy_features(y))
            
            # Espectrales
            features.update(self.extract_spectral_features(y))
            
            # Chroma
            features.update(self.extract_chroma_features(y))
            
            # Temporales
            features.update(self.extract_temporal_features(y))
            
            # Mel-espectrograma
            features.update(self.extract_mel_spectrogram_features(y))
            
            return features
        
        except Exception as e:
            print(f"Error procesando {audio_path}: {str(e)}")
            return None

def process_dataset(audio_dir, label_csv, output_path):
    """
    Procesa todo el dataset y guarda en formato parquet
    
    Args:
        audio_dir: Directorio con archivos .wav
        label_csv: CSV con las etiquetas (nombre_archivo, clase)
        output_path: Ruta donde guardar el archivo .parquet
    """
    print("Iniciando procesamiento del dataset...")
    
    # Cargar etiquetas
    labels_df = pd.read_csv(label_csv)
    print(f"Etiquetas cargadas: {len(labels_df)} archivos")
    print(f"Distribución de clases:\n{labels_df['clase'].value_counts()}\n")
    
    # Inicializar extractor
    extractor = AudioFeatureExtractor(sample_rate=16000)
    
    # Procesar cada archivo
    all_features = []
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    print(f"Procesando {len(audio_files)} archivos de audio...")
    
    for audio_file in tqdm(audio_files, desc="Extrayendo features"):
        features = extractor.extract_all_features(str(audio_file))
        if features is not None:
            all_features.append(features)
    
    # Crear DataFrame con features
    features_df = pd.DataFrame(all_features)
    
    # Unir con las etiquetas
    final_df = features_df.merge(
        labels_df, 
        left_on='nombre_archivo', 
        right_on='nombre_archivo',
        how='inner'
    )
    
    print(f"\nDataset final: {len(final_df)} muestras")
    print(f"Número de features extraídas: {len(final_df.columns) - 2}")  # -2 por nombre_archivo y clase
    print(f"\nDistribución final de clases:\n{final_df['clase'].value_counts()}\n")
    
    # Guardar en formato parquet
    output_file = Path(output_path) / "audio_features.parquet"
    final_df.to_parquet(output_file, index=False, compression='snappy')
    
    print(f"✓ Dataset guardado en: {output_file}")
    print(f"✓ Tamaño del archivo: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Mostrar algunas estadísticas
    print("\n" + "="*50)
    print("RESUMEN DE FEATURES EXTRAÍDAS:")
    print("="*50)
    print(f"- MFCCs: 60 features (20 coef × 3: mean, std, delta)")
    print(f"- Pitch: 5 features")
    print(f"- Energía (RMS, ZCR): 6 features")
    print(f"- Espectrales: ~15 features")
    print(f"- Chroma: 24 features")
    print(f"- Temporales: 2 features")
    print(f"- Mel-espectrograma: 4 features")
    print(f"\nTOTAL: {len(final_df.columns) - 2} features")
    
    return final_df

# RUTAS DE TU PROYECTO
if __name__ == "__main__":
    # Configurar rutas
    AUDIO_DIR = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\data\raw\Training_data"
    LABEL_CSV = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\data\raw\training_label.csv"
    OUTPUT_DIR = r"C:\Users\japal\Documents\TechCapital\Paralinguistic-Speech-Classification-for-Human-Vocalizations-2\model"
    
    # Crear directorio de salida si no existe
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Procesar dataset
    df = process_dataset(AUDIO_DIR, LABEL_CSV, OUTPUT_DIR)
    
    print("\n✓ Procesamiento completado exitosamente!")
    print(f"\nPrimeras filas del dataset:")
    print(df.head())
    
    print(f"\nInformación del dataset:")
    print(df.info())