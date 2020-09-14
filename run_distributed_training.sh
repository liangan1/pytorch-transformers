SQUAD_DATA=/lustre/dataset/SQuAD/
python lauch.py python -u ./examples/question-answering/run_squad.py     --model_type bert     --model_name_or_path bert-base-uncased     --do_train     --do_eval     --train_file ./glue_data/SQuAD/train-v1.1.json     --predict_file ./glue_data/SQuAD/dev-v1.1.json     --learning_rate 3e-5     --num_train_epochs 2     --max_seq_length 384     --doc_stride 128     --output_dir ../models/wsed_finetuned_squad/ --ipex --dnnl  --distributed

