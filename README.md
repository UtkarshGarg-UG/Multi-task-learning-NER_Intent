### Training

To train, use the following command: 

> !python train.py --task ner

Where task can have three values: ner, rel and mtl

* rel -> relation only model
* ner -> slot tagging only model
* mtl -> Multi task learning

### Testing

To test, use the following comman:

> !python test.py --file_path

* Default file path is 'data/hw1_test.xlsx'