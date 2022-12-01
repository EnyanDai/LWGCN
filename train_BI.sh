python -u train_BI.py --dataset Texas --layer=2 --hidden=128 --dropout=0.6 --lr=0.01 --weight_decay=0.005
python -u train_BI.py --dataset Wisconsin --layer=2 --hidden=128 --dropout=0.6 --lr=0.01 --weight_decay=0.005
python -u train_BI.py --dataset Cornell --layer=4 --hidden=32 --dropout=0.65 --lr=0.01 --weight_decay=0.005
python -u train_BI.py --dataset chameleon --layer=2 --hidden=64 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
python -u train_BI.py --dataset squirrel --layer=2 --hidden=256 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
python -u train_BI.py --dataset crocodile --layer=3 --hidden=256 --dropout=0.0 --lr=0.003 --weight_decay=0.0005
python -u train_DIS.py --dataset arxiv-year --layer 4 --hidden 256 --dropout 0.5 --weight_decay 0.0 --epochs=800

for seed in {10..12};
do
    python -u train_BI.py --dataset Cora --layer 64 --alpha 0.1 --weight_decay 1e-4 --seed=${seed} --dropout=0.6 --lr=0.01 --hidden=64
done

for seed in {10..12};
do
   python -u train_BI.py --dataset Citeseer --layer 32 --hidden 256 --lamda 0.6 --dropout 0.7 --alpha=0.1 --weight_decay 5e-4 --seed=${seed} --lr=0.01
done

for seed in {10..12};
do
    python -u train_BI.py --dataset Pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --weight_decay 5e-4 --seed=${seed} --lr=0.01
done 