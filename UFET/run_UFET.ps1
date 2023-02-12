$TIME_STAMP = Get-Date -Format o | ForEach-Object { $_ -replace ":", "." }
$SEED = 42 # 42, 32, 16 # 24, 8, 7 77

# Input and output file names
$RESULT_NAME = 'K_value_impact-' + $TIME_STAMP + '.txt'

$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'


foreach ($SEED in (42)) {
    foreach ($K in (80)){ 
        python parallel_TE_UFET.py `
            --train_batch_size 8 `
            --test_batch_size 128 `
            --num_train_epochs 5 `
            --learning_rate 1e-5`
            --threshold 0,0.02 `
            --max_seq_length 450 `
            --K $K `
            --N 1 `
            --embedding_method mean `
            --seed $SEED `
            --result_name $RESULT_NAME `
            --eval_each_epoch `
    ;
    }
}
