torch_ccl_path=$(python -c "import torch; import intel_pytorch_extension as ipex; import os;  print(os.path.abspath(os.path.dirname(ipex.__file__)))")
source $torch_ccl_path/../env/setvars.sh
export TOKENIZERS_PARALLELISM=1
python -m intel_pytorch_extension.launch --distributed --nproc_per_node=4 --nnodes=1 examples/question-answering/run_qa.py   --model_name_or_path bert-base-uncased   --dataset_name squad   --do_train   --do_eval   --per_device_train_batch_size 12   --learning_rate 3e-5   --num_train_epochs 2   --max_seq_length 384   --doc_stride 128   --output_dir /tmp/debug_squad/  --ipex --dnnl  --mix_precision --distributed_training --use_mpi_launcher 
