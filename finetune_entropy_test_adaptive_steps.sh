for i in {0..1500}
do
    echo "Number: $i"
    python finetune_entropy_test_adaptive_steps.py -n $i -a 1 -b 1 -m llava -t masking -u 4
done
