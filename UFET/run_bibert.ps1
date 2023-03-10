$TIME_STAMP = Get-Date -Format o | ForEach-Object { $_ -replace ":", "." }
$SEED = 42 # 42, 32, 16

# Input and output file names
$CODE_NAME = 'bi_bert.py'
$RESULT_NAME = 'result-' + $TIME_STAMP + '.txt'

$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'


$TIME_STAMP = Get-Date -Format o | ForEach-Object { $_ -replace ":", "." }

$STATEMENT_MODEL_NAME = 'model' + '-' + $TIME_STAMP + '.pth'
$LABEL_MODEL_NAME = 'model' + '-' + $TIME_STAMP + '.pth'

python $CODE_NAME `
    --train_batch_size 1 `
    --num_train_epochs 10 `
    --learning_rate 1e-5`
    --K 100,50,20 `
    --N_neg_sample 3000 `
    --embedding_method mean `
    --seed $SEED `
    --statement_model_save_path $STATEMENT_MODEL_NAME `
    --label_model_save_path $LABEL_MODEL_NAME `
    --result_name $RESULT_NAME `
    --ENABLE_WANDB `
    --eval_each_epoch `
    ;


$TIME_STAMP = Get-Date -Format o | ForEach-Object { $_ -replace ":", "." }

$STATEMENT_MODEL_NAME = 'model' + '-' + $TIME_STAMP + '.pth'
$LABEL_MODEL_NAME = 'model' + '-' + $TIME_STAMP + '.pth'

python $CODE_NAME `
    --train_batch_size 2 `
    --num_train_epochs 10 `
    --learning_rate 1e-5`
    --K 100,50,20 `
    --N_neg_sample 1500 `
    --embedding_method mean `
    --seed $SEED `
    --statement_model_save_path $STATEMENT_MODEL_NAME `
    --label_model_save_path $LABEL_MODEL_NAME `
    --result_name $RESULT_NAME `
    --ENABLE_WANDB `
    --eval_each_epoch `
    ;
