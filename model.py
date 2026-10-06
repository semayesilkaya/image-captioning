import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast


class CaptioningModel(nn.Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

        # 1. ViT ve GPT-2 Modellerini Önceden Eğitilmiş (Pretrained) Olarak İndir
        # Encoder (Kodlayıcı): ViT (Görüntüyü anlayacak)
        # Decoder (Çözücü): GPT-2 (Metin üretecek)
        print("Modeller Hugging Face üzerinden indiriliyor/yükleniyor...")
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "google/vit-base-patch16-224-in21k",
            "gpt2"
        )

        # 2. Tokenizer ve Görüntü İşlemcisi Ayarları
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # GPT-2'nin bir <pad> tokenı (boşluk doldurucu) yoktur, bu yüzden manuel eklememiz gerekir.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Cross-Attention'ı (Çapraz Dikkat) zorunlu kıl (GPT-2'nin görüntüyü görmesi için çok kritik!)
        self.model.config.add_cross_attention = True

    def forward(self, pixel_values, labels=None):
        # Modelin ileri yönlü (forward) çalışması
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs


if __name__ == "__main__":
    # Sınıfı çağır ve mimariyi ekrana bas
    print("Hibrit mimari kuruluyor...")
    caption_model = CaptioningModel()
    print("\n--- Model Kurulumu Başarılı! ---")

    # Modeldeki toplam eğitilebilir parametre sayısını görelim (Yüksek lisans tezleri için güzel bir detaydır)
    total_params = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
    print(f"Toplam Eğitilebilir Parametre Sayısı: {total_params:,}")