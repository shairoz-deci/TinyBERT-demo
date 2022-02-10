TinyBERT - Demo
======== 
Based on https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```
Useage
============
`distillation_demo.py` includes intermediate distillation on a proxy (random) dataset. To execute run:
```python
python -u /home/shai.rozenberg/PycharmProjects/TinyBERT-demo/distillation_demo.py --teacher_model /home/shai.rozenberg/PycharmProjects/TinyBERT-demo/models/teacher --student_model /home/shai.rozenberg/PycharmProjects/TinyBERT-demo/models/student
```
