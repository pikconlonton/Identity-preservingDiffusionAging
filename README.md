# 🧠 LIDA — LoRA-based Identity-preserving Diffusion Aging

> **LIDA** is a fine-tuned Stable Diffusion 1.5 pipeline that performs *age transformation* on human faces while *preserving identity consistency*.  
> It combines **Age-LoRA** and **Identity-LoRA** adapters with inpainting conditioning to achieve realistic, controllable, and identity-preserving facial aging results.

---

## 📘 Overview

Human faces naturally change over time, but existing AI models often fail to modify age while keeping identity intact.  
**LIDA (LoRA-based Identity-preserving Diffusion Aging)** solves this problem by leveraging **Dual-LoRA fine-tuning** over **Stable Diffusion 1.5**, trained separately for:

- 🧓 **Age-LoRA** → learns visual cues of different age groups (wrinkles, skin tone, facial volume).  
- 🪞 **Identity-LoRA** → learns personal facial features to preserve them across transformations.  

At inference, both adapters are combined with adjustable weights to generate realistic and consistent results.

---

## 🧩 Architecture


## 🧠 Model Pipeline
<p align="center">
  <img src="docs/Biểu đồ không có tiêu đề.drawio.png" width="600">
</p>


