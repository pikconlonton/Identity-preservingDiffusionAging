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

### 🔹 1. Data Preprocessing
- Generates **masks** and **captions** for each input image to provide detailed conditioning information.  
- Prepares data for **inpainting** and **cross-attention** fine-tuning.

### 🔹 2. Dual LoRA Modules
- **🧓 Age-LoRA:** Learns age-related patterns such as wrinkles, facial texture, and bone structure to enable realistic aging effects.  
- **🪞 Identity-LoRA:** Learns unique identity embeddings to preserve personal facial characteristics across transformations.  

> Both LoRA modules are fine-tuned independently and combined during inference for balanced and controllable outputs.

### 🔹 3. Stable Diffusion 1.5 Backbone
- Both LoRA adapters are integrated into the **cross-attention layers** of the **UNet** in Stable Diffusion 1.5.  
- Enables **joint conditioning** on *age* and *identity* during the **denoising process**, enhancing realism and identity consistency.

### 🔹 4. Inpainting Conditioning
- Combines the **original image**, **mask**, and **noise latent** \(z_t\) to perform localized edits.  
- Ensures smooth blending between modified (aged) regions and the unedited parts of the image.

### 🔹 5. Output
- Produces a high-fidelity, **age-modified image** that maintains both **identity integrity** and **visual realism**.

> **Figure:** Overview of the LIDA pipeline can be founded in "docs/Biểu đồ không có tiêu đề.drawio.png".

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/LIDA.git
cd LIDA
pip install -r requirements.txt
