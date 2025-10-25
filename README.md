# 🧠 LIDA — LoRA-based Identity-preserving Diffusion Aging

> **LIDA** is a fine-tuned Stable Diffusion 1.5 pipeline that performs *age transformation* on human faces while *preserving identity consistency*.  
> It combines **Age-LoRA** and **Identity-LoRA** adapters with inpainting conditioning to achieve realistic, controllable, and identity-preserving facial aging results.

---

## 📘 Overview

Human faces naturally change over time, but existing AI models often fail to modify age while keeping identity intact.  
**LIDA (LoRA-based Identity-preserving Diffusion Aging)** addresses this limitation by leveraging **Dual-LoRA fine-tuning** on top of **Stable Diffusion 1.5**, trained separately for:

- 🧓 **Age-LoRA** → Learns visual cues of different age groups (wrinkles, skin tone, facial volume).  
- 🪞 **Identity-LoRA** → Learns personalized facial embeddings to preserve appearance consistency.  

At inference time, both adapters are **merged with adjustable weights** to generate realistic and consistent results.

---

## 🧩 Model Architecture

The **LIDA** framework introduces a diffusion-based architecture that performs **age transformation** while **preserving facial identity** through dual LoRA fine-tuning on **Stable Diffusion 1.5**.

Given an **input image** and an **age prompt** (e.g., *“Person A at 60 years old”*), the model processes data through the following stages:

🔹 1. Data Preprocessing
- Generates **masks** and **captions** for each input image to provide detailed conditioning information.  
- Prepares data for **inpainting** and **cross-attention** fine-tuning.

🔹 2. Dual LoRA Modules
- **🧓 Age-LoRA:** Learns age-related patterns such as wrinkles, facial texture, and bone structure to enable realistic aging effects.  
- **🪞 Identity-LoRA:** Learns unique identity embeddings to preserve personal facial characteristics across transformations.  

> Both LoRA modules are fine-tuned independently and combined during inference for balanced and controllable outputs.

🔹 3. Stable Diffusion 1.5 Backbone
- Both LoRA adapters are integrated into the **cross-attention layers** of the **UNet** in Stable Diffusion 1.5.  
- Enables **joint conditioning** on *age* and *identity* during the **denoising process**, enhancing realism and identity consistency.

🔹 4. Inpainting Conditioning
- Combines the **original image**, **mask**, and **noise latent** \(z_t\) to perform localized edits.  
- Ensures smooth blending between modified (aged) regions and the unedited parts of the image.

🔹 5. Output
- Produces a high-fidelity, **age-modified image** that maintains both **identity integrity** and **visual realism**.

> **Figure:** Overview of the LIDA pipeline can be founded in "docs/ModelPipeline.drawio.png".

---

## ⚡ Quick Run (Summary)

🔹 1. Install dependencies:
   - pip install -r requirements.txt

🔹 2. (Colab) Mount Google Drive:
   - from google.colab import drive; drive.mount('/content/drive')

🔹 3. Prepare the dataset.
   - Dataset used for AgeLoRA in this link "https://github.com/JingchunCheng/All-Age-Faces-Dataset"
   - Dataset used for IdentityLoRA that you can prepared yourself with 20-30 self-images.
   - Create a dataset in the folder format of Diffusers==0.36.0.dev0.
AAF_LoRA_Dataset/
│
├── metadata.jsonl                ← text metadata file (one JSON record per image)
│
├── 00001.png                     ← training image #1
├── 00002.png                     ← training image #2
├── 00003.png                     ← training image #3
├── ...
│
└── (other .jpg / .png images)


🔹 4. Generate captions JSON (age buckets):
   -  Run the cell “Make ages_caption json” in src/CV_PTIT.ipynb → produces AAF_age_gender_caption_dataset.json.
     
🔹 5. Create dataset format for Diffusers:
   - Run the cell “Make age_lora_dataset…” → copy images into /content/AAF_LoRA_Dataset and generate metadata.jsonl

🔹 6. Train LoRA adapters:
   - accelerate launch examples/text_to_image/train_text_to_image_lora.py \
  --train_data_dir=/content/AAF_LoRA_Dataset \
  --output_dir=/content/drive/MyDrive/CV_PTIT/age_lora \
  ... (parameters: resolution, batch_size, lr, max_train_steps, rank, mixed_precision)

   - Identity-LoRA: trained similarly, but using the identity recognition dataset for --train_data_dir.

🔹 8. Inference & adapter combination:
   - Load pipeline, load_lora_weights for age and id, then pipe.set_adapters(["id","age"], adapter_weights=[1.0,0.2]); call pipe(prompt,...)

🔹 9. Inpainting (optional):
   - Making mask (edit white space) → Use StableDiffusionInpaintPipeline, load adapters, pipe(prompt, image=image, mask_image=mask, ...)
     
🔹 10. Experimentation & tips:
   - Adjust adapter_weights to balance identity preservation vs aging effect.
   - Use fp16 + xformers + bitsandbytes to save VRAM.
   - Always verify metadata.jsonl before training.
