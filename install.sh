conda create -n MultiPerson python=3.7 -y
conda activate MultiPerson

pip install torch==1.9.0 torchvision==0.10.0
pip install -r requirements.txt
pip install pyglet==1.5.27

git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git
mv pytorch-YOLOv4 YOLOv4