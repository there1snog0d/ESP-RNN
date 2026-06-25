# Нейроэволюционные стратегии Алгоритм ESP

**Вариант 3:** Алгоритм ESP | Полносвязная Рекуррентная сеть (1 скрытый слой) | Среда CartPole-v1

# Описание

Реализация алгоритма Enforced Sub-Populations (ESP) для обучения RNN-агента балансировке шеста. 
Проведено сравнение с Baseline (Standard ES), механистическая абляция, гиперпараметрические эксперименты и исследование робастности к шуму.

# Установка и запуск

1. Клонировать репозиторий

```bash
git clone https://github.com/there1snog0d/ESP-RNN.git
cd Table-agents-for-drone-navigation-problem
```

2. Создать виртуальное окружение
```bash
python -m venv venv
```

3. Активировать виртуальное окружение
```bash
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

4. Запуск обучения ESP
```bash
python train_esp.py
```

5. Запуск Baseline (Standard ES):
```bash
python train_baseline.py
```

6. Запуск экспериментов (Абляция и Робастность):
```bash
python experiments.py
```

7. Запуск гиперпараметрических экспериментов:
```bash
python hyperparams.py
```
