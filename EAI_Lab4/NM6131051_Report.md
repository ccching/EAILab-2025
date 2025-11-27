---
title: Lab4 - Homework Template

---

# Lab4 - Homework Template
### 1. About Knowledge Distillation (15%)
- What Modes of Distillation is used in this Lab ?
    - Offline distillation（老師先訓練好、固定不動，學生用老師的知識進行訓練）
- What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?
    - logits是神經網路最後一層的原始分數，代表模型對每個類別的confindent程度（還沒被換成機率）。也就是soft label的來源。
    -  公式：$softmax_T(logits) = \frac{exp(logit/T)}{\sum_j{exp(logit_j/T)}}$
    ，作用是調整輸出分布的平滑/銳利程度。
        - T越大：輸出分布更平滑，機率拉的更近。dark knowledge更容易被學生學到。
        - T越小：輸出分佈變尖銳。
        - 所以T設高一點，可以讓學生學到更多老師的細節，太低的話，學生只會學會標準答案
- In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?
    - 會從Teacher model的多個中間層feature map(隱藏層)提取知識。
### 2. Response-Based KD (30%)

- How you choose the Temperature and alpha?
    - 一開始就隨便亂選，後來上網有找一些資料說alpha設在0.5-0.8比較適合，T設5,10,50比較適合，所以就自己選了幾組參數下去訓練，不過差異都沒有很大，最終就採用ACC最高的那組。
    
        | No  | T     | alpha | Acc    |
        | --- | ----- | ----- | -------|
        | 1   | 10    |  0.8  | 88.91  |
        | 2   |  5    |  0.8  | 89.21  |
        | 3   | 50    |  0.8  | 86.19  |
        | 4   | 15    |  0.8  | 87.20  |
        | 5   | 4     |  0.5  | 88.49  |
- How you design the loss function?
    - 1.Student logits先用T做softmax再取log，產生soft機率。老師 logits 用 T 做 softmax。pytorch的KLDivLoss `F.kl_div(input,target)`，input需要給對數機率（log soft max），target則給機率本身（softmax）
    ```python=
    Student_prob = F.log_softmax(student_logits / T, dim=1)   
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    ```
    - 2.用KL Loss來衡量學生的預測分佈和老師的soft label分佈有多接近。最後補乘T^2做尺度修正，這樣loss大小才與溫度沒有關係。
     ```python=
    kd_loss = F.kl_div(Student_prob, teacher_prob, reduction='batchmean') * (T * T)
    ```
    - 3.Student output直接和正解label對比`F.cross_entropy(student_logits, labels)`
    - 4.混合loss 實作公式 $\mathcal{L}_{\text{Distill}} = \tau^2 \cdot \text{KLdiv}(Q_S, Q_T)$
    ```python=
    loss = alpha * kd_loss + (1 - alpha) * ce_loss
    ```

### 3. Feature-based KD (30%)

- How you extract features from the choosing intermediate layers?
    - 一開始先將student 和 teacher model的forward方法設計為回傳多層特徵（存成一個list），跟最終的分類結果一起回傳。在train得時候，呼叫模型後就能直接取得這些中間層特徵。
    ```python=
    student_logits, student_features = student(x)
    with torch.no_grad():
        teacher_logits, teacher_features = teacher(x)
    ```

- How you design the loss function?
    - 1.我採用講義的Normalized Feature Matching方法。normalize_feature function先去對每個feature map 的channel做L2正則化，讓每個位置的特徵直變成單位向量。
      $$normalize\_feature = \frac{feat}{||feat||_2+\epsilon}$$
    ```python=
     def normalize_feature(feat):
        norm = torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-8
        return feat / norm
    ```
    - 2.對每一組（student,teacher）中間層的特徵，先用adapter把student channel對齊到teacher channel，在分別做正則化。再用MSE做Loss計算，比較正則化後的student和teacher特徵。$$L_{feature} = \frac{1}{N}\sum_{i=1}^{N}MSE(normalize(s^{(i)},normalize(t^{(i)}))$$
    - 3.用Cross-Entropy Loss 讓學生能對齊標準答案，最後把normalized feature matching loss跟 CELoss按權重加總。

### 4. Comparison of student models w/ & w/o KD (5%)

|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.44     | 89.60     |
| Student from scratch       | 0.53     | 89.05     |
| Response-based student     | 1.85     | 88.91     |
| Featured-based student     | 0.26     | 87.16     |

### 5. Implementation Observations and Analysis (20%)
- Did any KD method perform unexpectedly? 
    - 從結果來看Response-based跟Feature-base的表現都比原本的模型還要差，預期應該是要好一些。我覺得可能的原因有幾個：
    - 1.T跟alpha沒有設定好：
        我這一個結果是用T = 10和aphla＝0.8。這個組合可能會讓學生過度專注於模仿老師的soft     labels，反而忽略了ground truth。也有嘗試過其他組合，帶目前嘗試的組合都跟from            scratch模型準確度差異不大，可能要再嘗試看看有沒有更好的組合。
    - 2.Student baseline已經很高了：
        我的student from scratch跟teacher的模型非常接近，只差了0.55，所以knowledge distillation帶來的提升空間感覺不大。