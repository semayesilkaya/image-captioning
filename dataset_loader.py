import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr30kDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        # Temizlediğimiz metinleri okuyoruz
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # Tezde taahhüt edilen: 224x224 boyutlandırma, kırpma ve veri artırımı
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Önce biraz büyük boyuta getir
            transforms.RandomCrop((224, 224)),  # Rastgele 224x224 kırp (veri artırımı için iyi bir yöntem)
            transforms.RandomHorizontalFlip(),  # Görüntüyü rastgele yatay çevir (Data augmentation)
            transforms.ToTensor(),  # PyTorch Tensörüne (matrise) çevir
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standart ViT normalizasyonu
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Resmin ismini al (örn: 1000092795.jpg)
        img_name = str(self.df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_name)

        # Resmi yükle ve RGB formatında olduğundan emin ol
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Hatalı veya eksik fotoğraf: {img_path}")
            # Hata olursa yerine siyah bir tensor dön (sistemin çökmemesi için)
            image = Image.new('RGB', (224, 224))

        # Ön işleme adımlarını (224x224 kırpma vb.) uygula
        image = self.transform(image)

        # Temizlenmiş altyazıyı al
        caption = str(self.df.iloc[idx, 1])

        return image, caption


# Kodun doğru çalışıp çalışmadığını test etmek için:
if __name__ == "__main__":
    csv_yolu = 'dataset/processed_captions.csv'
    resim_klasoru = 'dataset/Images/'

    # Sınıfı çağırıyoruz
    test_dataset = Flickr30kDataset(csv_file=csv_yolu, img_dir=resim_klasoru)

    # Veri setinden 0. (ilk) elemanı çekiyoruz
    ilk_resim_tensoru, ilk_altyazi = test_dataset[0]

    print(f"Test Başarılı!")
    print(f"Çekilen Altyazı: {ilk_altyazi}")
    print(f"Görselin Tensor Boyutu (Kanal x Yükseklik x Genişlik): {ilk_resim_tensoru.shape}")