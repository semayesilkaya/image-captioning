import torch
from PIL import Image
from model import CaptioningModel
from dataset_loader import transforms


def generate_caption(image_path):
    print("Model yükleniyor...")
    # Şimdilik modelin o anki ağırlıklarını yüklüyoruz
    # (İleride buraya saatlerce eğitilip 'kaydedilmiş' modeli yükleyeceğiz)
    model_container = CaptioningModel()
    model = model_container.model
    tokenizer = model_container.tokenizer

    # Modeli değerlendirme (test) moduna al
    model.eval()

    print(f"Fotoğraf okunuyor: {image_path}")
    # Fotoğrafı aç ve modelin anlayacağı Tensör (224x224) formatına çevir
    image = Image.open(image_path).convert('RGB')

    # Sadece boyutlandırma ve tensör yapma (Test sırasında veri artırımı / kırpma yapılmaz)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pixel_values = transform(image).unsqueeze(0)  # Batch boyutu ekle (1, 3, 224, 224)

    print("Yapay zeka fotoğrafı inceliyor ve metin üretiyor...")
    # GPT-2'ye "başla" komutunu ver ve maksimum 30 token (kelime) uzunluğunda üretmesini söyle
    generated_ids = model.generate(
        pixel_values,
        max_length=30,
        num_beams=4,  # Daha mantıklı cümleler kurması için beam search
        decoder_start_token_id=tokenizer.bos_token_id
    )

    # Üretilen matematiksel ID'leri tekrar İngilizce kelimelere çevir
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 50)
    print(f"SONUÇ ALTYAZI: {generated_caption}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Test etmek için dataset'in içinden rastgele bir fotoğrafın yolunu yazalım
    # Klasöründe var olan herhangi bir fotoğrafın ismini buraya yazabilirsin
    test_resmi = 'dataset/Images/1000268201.jpg'
    generate_caption(test_resmi)