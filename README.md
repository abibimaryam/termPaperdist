# 📑 Название | Title

Инициализация весов трансформеров на основе свёрточных нейросетей с последующим дообучением
Initialization of transformer weights based on convolutional neural networks, followed by further training

## 🧠 Описание | Description

Проект реализует гибридную архитектуру нейронной сети на основе ResNet18, где сверточные блоки заменены на блоки TransformerConvBlock. Цель — использовать инициализацию весов из сверточных нейросетей через послойную дистилляцию для трансформеров и дообучить модель  fine-tuning.

The project implements a hybrid Transformer-CNN architecture where ResNet18 blocks are replaced with custom TransformerConvBlocks. We initialize transformer weights from CNN using layer-wise distillation and fine-tune them.

---

## 📁 Структура проекта | Project Structure

- `main.py` — запуск процесса послойной дистилляции
- `finetune.py` — дообучение модели после дистилляции
- `transformer.py` — описание архитектуры TransformerConvBlock и модели
- `data/` — входные данные (CIFAR-100)
- `distillation_log.txt` — лог послойной дистилляции
- `training_log.txt` — лог обучения
- `dist_model_transformer.pth` — модель после дистилляции
- `student_finetuned.pth` — итоговая модель после дообучения

---

## ⚙️ Установка | Setup

```bash
git clone <repository_url>
pip install -r requirements.txt
```

---

## 🚀 Использование | Usage

```bash
python main.py --config config.yaml
python finetune.py --model_path dist_model_transformer.pth
```

---

## 📊 Методология | Methodology

- **Послойная дистилляция**: каждое соответствие слоёв teacher/student обучается отдельно с использованием MSELoss между фичами
- **Fine-tuning**: полное дообучение модели с CrossEntropyLoss и аугментацией данных
- **Оптимизаторы**: AdamW и CosineAnnealingLR

---

## 📈 Результаты | Results

| Компонент              | ResNet-18 | Transformer | Экономия |
| ------------------------------- | --------- | ----------- | ---------------- |
| Всего параметров | 11.2M     | 6.2M        | 45%              |
| Точность (Accuracy)     | 79.00%    | 29.00%      |                  |

---

## 🧾 Лицензия | License

Этот проект распространяется под лицензией MIT.

---

## 🙏 Благодарности | Acknowledgements

- [An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929)
- [Timm library](https://github.com/rwightman/pytorch-image-models)
- [Transfer learning](https://habr.com/ru/companies/skillfactory/articles/835020/)
