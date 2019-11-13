###
### script for BERT inference on cpu
### reference:
###   https://github.com/mingfeima/pytorch-transformers#run_gluepy-fine-tuning-on-glue-tasks-for-sequence-classification
###
### 1. prepare dataset:
###   https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
###
### 2. install:
###   pip install --editable .
###   pip install -r ./examples/requirements.txt
###
### 3. use run_perf:
###   ./run_inference_cpu.sh --run_perf (throughput)
###   ./run_inference_cpu.sh --single --run_perf (realtime) 

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


PREFIX=""
ARGS=""
BATCH_SIZE=8
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  TOTAL_CORES=4
  LAST_CORE=`expr $TOTAL_CORES - 1`
  PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
  shift
fi

if [[ "$1" == "--run_perf" ]]; then
  echo "### using run_perf"
  ARGS="--run_perf"
fi

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using ARGS=$ARGS\n"
for task in "QNLI" "MRPC" 
do

GLUE_DIR=/lustre/dataset/glue_data
TASK_NAME=${task}

OUTPUT=${GLUE_DIR}/weights/${TASK_NAME}_output/
if [[ -d "$OUTPUT" ]]; then
  echo "### using model file from $OUTPUT"
else
  echo -e "\n### model file not found, run fune tune first!\n###  ./run_training_gpu.sh\n"
  exit
fi

$PREFIX python ./examples/run_glue.py --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name ${TASK_NAME} \
    --do_eval \
    --do_lower_case \
    --do_calibration \
    --data_dir $GLUE_DIR/$TASK_NAME/ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size $BATCH_SIZE \
    --no_cuda \
    --output_dir $OUTPUT $ARGS
done
#echo -e "\n### samples/sec = batch_size * it/s\n### batch_size = $BATCH_SIZE"
