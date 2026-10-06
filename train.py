import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # İlerleme çubuğu için

from dataset_loader import Flickr30kDataset
from model import CaptioningModel


def collate_fn(batch):
    # DataLoader içindeki tensörleri tek bir "paket" haline getirir
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    return images, captions


def train_model():
    print("Donanım kontrol ediliyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak Donanım: {device}")

    # 1. Modeli ve Veri Setini Yükle
    model_container = CaptioningModel()
    model = model_container.model.to(device)
    tokenizer = model_container.tokenizer

    csv_yolu = 'dataset/processed_captions.csv'
    resim_klasoru = 'dataset/Images/'

    print("Veri seti yükleniyor...")
    # Sadece ilk 500 veriyi alarak mini bir test eğitimi yapalım (Tamamı için kodu güncelleyeceğiz)
    full_dataset = Flickr30kDataset(csv_file=csv_yolu, img_dir=resim_klasoru)
    subset_dataset = torch.utils.data.Subset(full_dataset, range(500))

    # Bilgisayarın belleğini zorlamamak için batch_size=4 (Aynı anda 4 resim işler)
    dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 2. Optimizasyon Ayarları (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("\n--- Eğitim (Training) Başlıyor ---")
    model.train()

    for epoch in range(1):  # Şimdilik 1 döngü
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for images, captions in progress_bar:
            images = images.to(device)

            # GPT-2'nin beklediği metinleri (caption) tokenlara çeviriyoruz
            encoded_captions = tokenizer(
                captions, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).input_ids.to(device)

            # Optimizatörü sıfırla
            optimizer.zero_grad()

            # İleri Yönlü Geçiş (Forward)
            outputs = model(pixel_values=images, labels=encoded_captions)
            loss = outputs.loss

            # Geriye Yayılım (Backpropagation)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'Loss (Hata)': f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} Tamamlandı. Ortalama Kayıp (Loss): {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    train_model()