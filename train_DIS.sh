# python -u train_DIS.py --dataset Texas --layer=2 --hidden=32 --dropout=0.5 --lr=0.01 --weight_decay=0.005
# python -u train_DIS.py --dataset Wisconsin --layer=2 --hidden=64 --dropout=0.5 --lr=0.01 --weight_decay=0.005

# python -u train_DIS.py --dataset chameleon --layer=2 --hidden=64 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
# python -u train_DIS.py --dataset squirrel --layer=2 --hidden=64 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
layers=(4 5 6)
for layer in ${layers[@]}
do
    python -u train_DIS.py --dataset crocodile --layer=${layer} --hidden=256 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
done