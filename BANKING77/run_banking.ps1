$TIME_STAMP = Get-Date -Format o | ForEach-Object { $_ -replace ":", "." }
$SEED = 42 # 42, 32, 16 # 24, 8, 7 77

# Input and output file names
$RESULT_NAME = 'k_impact_three-shot-' + $TIME_STAMP + '.txt'

$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

foreach ($FILE in ('one_shot_0', 'one_shot_1', 'one_shot_2', 'three_shot_0', 'tree_shot_1', 'three_shot_2', 'five_shot_0', 'five_shot_1', 'five_shot_2')){
    foreach ($SEED in (42)) {
        foreach ($K in (25)){ 
            python parallel_TE_BANKING77 `
                --train_file $FILE `
                --train_batch_size 8 `
                --test_batch_size 512 `
                --num_train_epochs 5 `
                --learning_rate 5e-6`
                --max_seq_length 300 `
                --K $K `
                --T 1 `
                --embedding_method mean `
                --seed $SEED `
                --result_name $RESULT_NAME `
                --eval_each_epoch `
        ;
        }
    }
}
