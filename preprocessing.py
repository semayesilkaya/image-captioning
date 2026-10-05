import pandas as pd
import string

file_path = 'dataset/captions.txt'

# 1. Veriyi okuma
df = pd.read_csv(file_path, delimiter=',')
df.columns = df.columns.str.strip() # Sütun isimlerindeki olası boşlukları temizle

# Sadece string olan veriler üzerinde işlem yapacağımızdan emin olalım
df['caption'] = df['caption'].astype(str)

# 2. Küçük harfe dönüştürme
df['caption'] = df['caption'].str.lower()

# 3. Noktalama işaretlerini temizleme
df['caption'] = df['caption'].str.replace(f'[{string.punctuation}]', '', regex=True)

# Fazladan oluşan yan yana boşlukları tek boşluğa düşürüp baş/son boşlukları kırpalım
df['caption'] = df['caption'].str.replace(r'\s+', ' ', regex=True).str.strip()

# 4. <start> ve <end> özel belirteçlerini (token) ekleme
df['caption'] = '<start> ' + df['caption'] + ' <end>'

# Sonuçları kontrol etmek için ilk 5 satırı ekrana yazdır
print("\n--- İlk 5 Satırın İşlenmiş Hali ---")
print(df.head())

# 5. Temizlenmiş veriyi yeni bir dosya olarak kaydetme
output_path = 'dataset/processed_captions.csv'
df.to_csv(output_path, index=False)
print(f"\nİşlem tamamlandı! Toplam {len(df)} satır '{output_path}' konumuna kaydedildi.")