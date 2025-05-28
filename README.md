# 🐘 AI for Elephant Conservation: Spotting Viral Gold with HL-CLIP 🎞️✨

**Transforming countless hours of wildlife footage into captivating, shareable moments to champion elephant conservation.**

## The Challenge: Finding the "Wow" in the Wild 🌾

Elephants are magnificent, but how do we get the world to pay more attention to their plight and the conservation efforts underway? While researchers gather vital data, a parallel opportunity exists: identifying those truly special, awe-inspiring, or heartwarming moments in video footage that can go viral. These "viral gold" clips can significantly boost public awareness, drive engagement, and attract support for elephant conservation. Manually sifting through terabytes of video for these gems is like finding a needle in a haystack – an impossible task for busy conservation teams.

## Our Solution: Your AI Content Scout & Highlight Reel Creator 💡

This project introduces an AI-powered system to automatically detect and highlight potentially viral segments in elephant video footage. We leverage a **CLIP-based highlight detection** model, fine-tuning a powerful pre-trained vision-language model to understand and identify engaging and significant events. Think of it as an "AI Content Scout" that:

*   **Pinpoints Viral Potential:** Identifies unique behaviors, charming interactions, dramatic scenes, or simply beautiful shots that are likely to captivate an online audience.
*   **Accelerates Content Creation:** Drastically reduces the time needed to find compelling footage for awareness campaigns, social media, and documentaries.
*   **Boosts Engagement:** Helps create a steady stream of engaging content to keep elephant conservation in the public eye and foster a global community of supporters.
*   **Amplifies Conservation Stories:** Enables conservationists to more easily share the narratives of the elephants they work to protect, making the cause more relatable and urgent.

We've journeyed through data preparation, model training, rigorous evaluation, and advanced optimization techniques (including pruning and FP16 compilation with TensorRT) to create a practical, effective, and environmentally conscious AI solution. This project is submitted to the [Moodeng: AI for Social Good - Elephant Challenge](https://moodeng.media.mit.edu/).

---

## 🏆 Hackathon Submission Checklist & Details

This project fulfills the requirements for the AI for Elephants Hackathon:

### 1. 🌐 All Projects Must Be Open-Source

*   **Repository:** This entire project, including all code, model weights (for the fine-tuned, pruned, and FP16 TensorRT versions), and documentation, is publicly available at: [https://github.com/ap9055097/VideoHilightModel](https://github.com/ap9055097/VideoHilightModel)
*   **License:** This project is licensed under the MIT License (see `LICENSE` file).

### 2. 🧠 AI Model/System Prototypes

*   **Model Submission:**
    *   The fine-tuned HL-CLIP model (`best_hlclip_model.pth`).
    *   The pruned fine-tuned HL-CLIP model (`best_prune_hlclip_model.pth`).
    *   The FP16 TensorRT compiled model (`trt_hlclip_fp16.ts`).
    *   These are available in the repository [e.g., in a `models/` directory or via a release].
*   **Model Card:** This `README.md` serves as the primary model card.
    *   **Model Structure & Methodology:**
        *   **Base Model:** OpenAI's CLIP (ViT-B/32 architecture). See: [Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020).
        *   **Highlight Detection Adaptation:** We adopted principles for adapting CLIP to video highlight detection, inspired by works like [Unleash the Potential of CLIP for Video Highlight Detection (Han et al., 2024)](https://arxiv.org/abs/2404.01745). This involved fine-tuning the CLIP visual encoder. Specifically, the last 2 transformer layers of the visual encoder were unfrozen and trained, while the rest of the pre-trained weights were kept frozen. A custom classification head (comprising a Linear layer, ReLU activation, and an output Linear layer with Sigmoid) was added to predict frame-level highlight scores. Our specific implementation is detailed in the provided notebook and referred to as `HLCLIPModel`.
        *   **Intended Use Case:** To automatically identify and flag semantically significant and potentially "viral-worthy" segments in videos featuring elephants. This assists in quickly locating footage that can be used for public engagement, storytelling, and raising awareness for conservation. The model outputs per-frame highlight probabilities, which are post-processed to define highlight segments.

### 3. 🎬 Video Demonstration

*   A short, public-friendly video explaining our solution, its goals (focused on viral content for awareness), and potential impact on elephant conservation is available here:
    `[LINK_TO_YOUR_YOUTUBE/VIMEO_VIDEO_DEMO]`
    *(The video should briefly cover the problem of finding engaging footage, how our CLIP-based model helps, show some example predictions of "viral-worthy" moments, and discuss how this can boost conservation awareness.)*

### 4. 📜 Scientific Report

*   A document covering the project’s motivation (finding engaging "viral" content for elephant conservation awareness), detailed methodology, dataset description, training procedures, evaluation results (including comparisons), and discussion can be found here:
    *   **Key References:**
        *   Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. *arXiv preprint arXiv:2103.00020*.
        *   Han, D., Seo, S., Park, E., Nam, S. U., & Kwak, N. (2024). Unleash the Potential of CLIP for Video Highlight Detection. *arXiv preprint arXiv:2306.00749*.
    *(This report would expand significantly on the details presented in this README and include further analysis as per standard scientific reporting.)*

### 5. 🌍 AI Model Footprint on Environment

We are committed to responsible AI development and have taken several steps to consider and minimize the environmental footprint of our model:

*   **1. Leveraging Pre-trained Models (Transfer Learning):**
    *   **Effort:** Our core approach utilizes OpenAI's pre-trained CLIP model. This saves an immense amount of computational resources and associated carbon emissions that would have been required to train such a large vision-language model from scratch. We only fine-tuned a small portion of the network.
    *   **Impact:** This is the single most significant factor in reducing our training carbon footprint.

*   **2. Model Pruning:**
    *   **Effort:** We implemented L1 Unstructured global pruning on the fine-tuned model, targeting a 20% reduction in weights across linear and convolutional layers, followed by a short re-fine-tuning phase.
    *   **Impact:** Pruning reduces the model's parameter count, which can lead to:
        *   Smaller model size on disk (less storage energy).
        *   Potentially faster inference (fewer operations), thus lower inference energy consumption per video. Our results show that pruning maintained, and even slightly improved, the F1-score.
    *   **Details:** Pruning targeted 39 layers, including the custom head and unfrozen CLIP layers.

*   **3. PyTorch JIT Compilation (TorchScript):**
    *   **Effort:** The pruned, fine-tuned model was converted to TorchScript using `torch.jit.trace`.
    *   **Impact:** TorchScript can optimize the model graph and enable execution in more efficient C++ runtimes. This primarily benefits inference speed and can reduce CPU/GPU utilization per inference, thereby lowering energy consumption during deployment.

*   **4. FP16 Quantization with Torch-TensorRT:**
    *   **Effort:** The TorchScript model was successfully compiled using Torch-TensorRT with `torch.float16` precision.
    *   **Impact:** FP16 precision significantly reduces model size and computational requirements during inference compared to FP32, leading to faster inference and lower energy consumption on compatible hardware (like NVIDIA T4 GPUs). This makes deployment more efficient and environmentally friendlier. The compiled model is saved as `trt_hlclip_fp16.ts`.

*   **5. Efficient Training & Development Practices:**
    *   **Dataset Subsampling:** During initial development and hyperparameter exploration, we worked with a smaller subset of the full dataset (`num_train_samples = 10 good + 10 bad videos`) to iterate quickly and avoid unnecessary computation on the full dataset until the pipeline was stable.
    *   **Limited Epochs:** We trained for a limited number of epochs (e.g., 4 epochs for initial fine-tuning and pruning re-training) sufficient to demonstrate learning and improvement, avoiding excessive training.
    *   **Colab GPU Resources:** Training was performed on Google Colab using available T4 GPUs, which are relatively power-efficient.

*   **6. Consideration of Further Quantization (Future Work for Carbon Reduction):**
    *   **Potential Impact:** While FP16 is a significant step, INT8 quantization via TensorRT or PyTorch's native static/dynamic quantization could offer further reductions in model size and inference energy. This remains a potential next step for even lower-carbon deployment.

*   **Documentation on Energy Use (Qualitative & Efforts-Based):**
    *   Direct, precise energy measurement (kWh) is challenging in a shared cloud environment like Google Colab without specialized tools.
    *   Our sustainability efforts focused on *reducing computational load* through the techniques listed above.
    *   The reduction in FLOPs, parameters, and the use of lower precision (FP16) are indirect indicators of reduced energy needs per inference.
    *   **Future Steps for Quantification:** If deployed, tools like CodeCarbon could be integrated, or power measurements could be taken on dedicated hardware to quantify energy savings more precisely.

---

## 🛠️ How We Did It: The Journey

### 1. Data Preparation & Preprocessing

*   **Dataset Source:** Videos were sourced from Google Drive. The primary dataset used for this project can be accessed here: [Elephant Highlight Video Dataset](https://drive.google.com/drive/folders/1M9yBS0Q_0iLKzOIzYcXHODTat1e58u9d?usp=drive_link). We used separate `train_augmented` and `val` directories within a structure like `/content/drive/MyDrive/elephant_clip/`.
*   **Annotation Strategy:**
    *   A simple yet effective annotation strategy was used: videos placed in a subdirectory containing `/bad/` in its path (e.g., `val/bad/video.mp4`) were treated as having no specific highlight segments (empty annotation list `[]` for their `video_id`). These represent non-viral or standard footage.
    *   Videos in other paths (e.g., `val/good/video.mp4` or top-level in `train_augmented`) for which no explicit annotation was provided in `annotations_data` were considered as **entirely highlight footage**. These represent potentially viral or highly engaging content.
    *   For training, we initially sampled `10` "good" (assumed full highlight) and `10` "bad" (no highlight) videos to create a balanced small dataset for faster iteration.
*   **Frame Extraction:** Videos were processed using OpenCV (`cv2.VideoCapture`). Frames were extracted at a specified `stride`.
*   **Preprocessing for CLIP:** Each extracted frame (as a PIL Image) was passed through the standard `preprocess` function provided by OpenAI's CLIP model, which includes resizing to 224x224, normalization, and conversion to a PyTorch tensor.

### 2. Model Architecture: HL-CLIP Style

*   **Base:** We started with the ViT-B/32 (Vision Transformer) visual encoder from OpenAI's CLIP model.
*   **Fine-tuning Strategy:**
    *   The **last 2 transformer `resblocks`** of the visual encoder were unfrozen.
    *   A **custom classification head** was added: `Linear(visual_output_dim, 512) -> ReLU -> Linear(512, 1) -> Sigmoid`.

### 3. Training

*   **Device:** GPU (NVIDIA T4 on Google Colab).
*   **Optimizer:** `torch.optim.AdamW` (lr `1e-5`).
*   **Loss Function:** `torch.nn.BCELoss`.
*   **Data Loading:** Custom `ElephantHighlightDataset` and `DataLoader` with a custom `collate_fn`.
*   **Epochs:** 4 epochs for initial fine-tuning and pruning re-training.

### 4. Evaluation & Results

We performed several evaluation steps:

*   **Validation Set:** A separate set of videos (`val_video_paths`, `val_annotations_data`) was used.
*   **Metrics:** Frame-level Precision, Recall, F1-Score, and Accuracy.
*   **Optimal Threshold Finding:** Iterated through thresholds to maximize F1-score.

**Key Results Summary (metrics based on the model *before* JIT/TensorRT compilation for speed/size optimization):**

| Model                          | Optimal Threshold | Precision | Recall | F1-Score | Accuracy |
| :----------------------------- | :---------------: | :-------: | :----: | :------: | :------: |
| Zero-Shot CLIP                 |      0.2329       |  0.3191   | 1.0000 |  0.4839  |  0.3333  |
| Fine-tuned HL-CLIP             |      0.5049       |  0.3913   | 0.9000 |  0.5455  |  0.5312  |
| Pruned Fine-tuned HL-CLIP      |      0.4951       |  0.4590   | 0.9333 |  0.6154  |  0.6354  |

*(Note: TN/FP/FN/TP are from the validation run with the optimal threshold for each respective model before deployment optimizations. The FP16 TensorRT model aims for efficiency while preserving these metrics as closely as possible.)*

**Observations:**

*   **Fine-tuning Works:** The fine-tuned model (F1: 0.5455) significantly outperformed Zero-Shot CLIP.
*   **Pruning is Effective:** The Pruned model achieved the **highest F1-score (0.6154)**.
*   **Zero-Shot Challenges:** Zero-Shot CLIP struggled with precision for this specific "viral highlight" task.
*   **JIT & TensorRT FP16:** These steps optimize the pruned model for efficient deployment, as shown in the notebook.

*(Score distribution plots from `matplotlib` in the notebook visually support these findings.)*

### 5. Optimization Techniques Applied

*   **Pruning:** L1 Unstructured global pruning (20%).
*   **PyTorch JIT Compilation (TorchScript):** Model traced with `torch.jit.trace`.
*   **FP16 Quantization with Torch-TensorRT:** The TorchScript model was compiled to FP16 precision.

---

## 🚀 How to Use This Model

The primary way to use this model, from data loading and preprocessing to training, evaluation, and prediction with the optimized models, is by following the **`[HL_CLIP]_elephant_video_training_v0_0_3_.ipynb`** notebook provided in this repository: [https://github.com/ap9055097/VideoHilightModel](https://github.com/ap9055097/VideoHilightModel).

Here's a general guide:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/ap9055097/VideoHilightModel.git
    cd VideoHilightModel
    ```

2.  **Set up Your Environment:**
    *   It's highly recommended to use Google Colab with a GPU runtime (like the T4 used for development) to run the notebook. This ensures most dependencies are pre-configured or easily installable via `!pip` commands within the notebook.
    *   If running locally, create a virtual environment and install all necessary packages. Key packages include `torch`, `torchvision`, `torchaudio`, `ftfy`, `regex`, `tqdm`, `opencv-python`, `scikit-learn`, `numpy`, `matplotlib`, and `torch-tensorrt==2.2.0`. You will also need to install CLIP directly from GitHub: `pip install git+https://github.com/openai/CLIP.git`. Ensure your CUDA and PyTorch versions are compatible with Torch-TensorRT. The notebook includes installation cells for these.

3.  **Prepare Data & Model Weights:**
    *   **Dataset:**
        *   Download the dataset from [Elephant Highlight Video Dataset](https://drive.google.com/drive/folders/1M9yBS0Q_0iLKzOIzYcXHODTat1e58u9d?usp=drive_link).
        *   Mount your Google Drive in Colab (`from google.colab import drive; drive.mount('/content/drive')`).
        *   Update the paths in the notebook (e.g., `directory_to_search = "/content/drive/MyDrive/elephant_clip/train_augmented"`) to point to your dataset location if it differs.
    *   **Model Weights:**
        *   The notebook saves model weights like `best_hlclip_model.pth`, `best_prune_hlclip_model.pth`, and `trt_hlclip_fp16.ts` during its execution. These will be created in your Colab environment's working directory.
        *   If you are skipping training and want to use pre-trained weights provided in this repository (if available as releases or in a `models/` folder), download them and ensure the paths in the prediction/evaluation sections of the notebook point to these downloaded files.

4.  **Run the Notebook:**
    *   Open `[HL_CLIP]_elephant_video_training_v0_0_3_.ipynb` in Google Colab or your local Jupyter environment.
    *   Execute the cells sequentially by following the instructions and comments within the notebook.
    *   **Data Loading & Preprocessing:** Cells under "Load Dataset List" and "Prepare Dataset" handle this.
    *   **Training:** Cells under "Training" will fine-tune the model.
    *   **Evaluation:** Cells under "Metrics", "Zero shot model", "Zero-shot VS Fine-Tuning", and "Find Best Threshold" perform various evaluations.
    *   **Optimization:** Cells under "Pruning", "PyTorch JIT Compilation (TorchScript)", and "Quantization" (which leads to the FP16 TensorRT model) optimize the model.
    *   **Prediction (Inference):**
        *   The section "**Predict**" in the notebook demonstrates how to use the `predict_highlights` function.
        *   This section shows loading the `best_hlclip_model.pth` and then later, in the "PyTorch JIT Compilation (TorchScript)" and "Quantization" sections, it demonstrates running predictions with the `traced_model` and `loaded_trt_model` (the FP16 TensorRT version).
        *   To predict on your own videos, modify the `val_video_paths[...]` in these prediction cells or provide a direct path to your video file. Ensure the chosen model (e.g., `model`, `prune_fine_tuned_model`, or `loaded_trt_model`) and its corresponding optimal threshold (found during evaluation) are used in the `predict_highlights` function.

        Example of how prediction is called in the notebook (using the FP16 TensorRT model):
        ```python
        # In the notebook, 'loaded_trt_model' is the loaded FP16 TensorRT model.
        # 'preprocess' is the CLIP preprocessing function.
        # 'device' is set (e.g., "cuda").
        # 'val_video_paths' is an example video path from the validation set.
        # The optimal threshold for the pruned model was ~0.4951, used as an example here.
        
        highlights = predict_highlights(val_video_paths, loaded_trt_model, preprocess, device, threshold=0.4951) 
        print(f"Detected highlights in {val_video_paths}: {highlights}")
        ```
        Refer to the notebook for the complete context and function definitions.

---

## 🌱 Future Work & Enhancements

*   **Expand & Diversify Dataset:** Incorporate a wider array of elephant videos with more nuanced annotations (e.g., specific "viral" actions, emotional cues) to refine the model's understanding of engaging content.
*   **Advanced Augmentations:** Implement video-specific augmentations.
*   **Segment-Level mAP:** Evaluate using segment-level metrics for highlight detection.
*   **INT8 Quantization:** Fully implement INT8 quantization with TensorRT for maximum efficiency and minimal environmental footprint.
*   **User-Friendly Interface:** Develop a simple tool for conservationists and content creators to easily upload videos and get back highlight suggestions, perhaps with an editor to refine them.
*   **Content-Type Classification:** Extend the model to not just find highlights, but also classify their potential "viral" category (e.g., "funny," "touching," "action-packed," "educational").

---

## 🙏 Acknowledgements

*   OpenAI for the pre-trained CLIP model.
*   The authors of research papers that explore adapting CLIP for video tasks, such as Han et al. (2024).
*   The organizers and sponsors of the [Moodeng: AI for Social Good - Elephant Challenge](https://moodeng.media.mit.edu/).

---
