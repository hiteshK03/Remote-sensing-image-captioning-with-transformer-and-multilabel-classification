import argparse
import os
import torch
import pandas as pd

from fairseq import options, tasks, checkpoint_utils
from fairseq.data import encoders
from tqdm import tqdm

import data

classes = ['airplane','bare-soil','buildings','cars','chaparral','court','dock','field','grass','mobile-home','pavement','sand','sea','ship','tanks','trees','water']

def main(script_args, model_args):
    split = script_args.split
    os.path.join(model_args.features_dir, )
    predictions_cap = predict(image_id_path=os.path.join(model_args.captions_dir, f'{split}-ids.txt'),
                          grid_features_path=os.path.join(model_args.features_dir, f'{split}-features-grid'),
                          obj_features_path=os.path.join(model_args.features_dir, f'{split}-features-obj'),
                          obj_features_meta_path=os.path.join(model_args.features_dir, f'{split}-features-obj', 'metadata.csv'),
                          model_args=model_args,
                          mode='captions')

    predictions_label = predict(image_id_path=os.path.join(model_args.captions_dir, f'{split}-ids.txt'),
                          grid_features_path=os.path.join(model_args.features_dir, f'{split}-features-grid'),
                          obj_features_path=os.path.join(model_args.features_dir, f'{split}-features-obj'),
                          obj_features_meta_path=os.path.join(model_args.features_dir, f'{split}-features-obj', 'metadata.csv'),
                          model_args=model_args,
                          mode='labels')    

    if not script_args.no_console_output:
        print_predictions(predictions_cap, mode='captions')
        print_predictions(predictions_label, mode='labels')

    if script_args.output:
        store_predictions_as_csv(predictions_cap, script_args.output, mode='caption')
        store_predictions_as_csv(predictions_label, script_args.output_l, mode='labels')

def predict(image_id_path: str,
            grid_features_path: str,
            obj_features_path: str,
            obj_features_meta_path: str,
            model_args,
            mode) -> pd.DataFrame:
    print(model_args)
    use_cuda = torch.cuda.is_available() and not model_args.cpu

    task = tasks.setup_task(model_args)
    captions_dict = task.target_dictionary

    models, _model_args = checkpoint_utils.load_model_ensemble(model_args.path.split(':'), task=task)

    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if model_args.no_beamable_mm else model_args.beam,
            need_attn=model_args.print_alignment,
        )

        if torch.cuda.is_available() and not model_args.cpu:
            model.cuda()

    generator = task.build_generator(model_args)
    tokenizer = encoders.build_tokenizer(model_args)
    bpe = encoders.build_bpe(model_args)

    def decode(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    sample_ids = data.read_image_ids(model_args.input, non_redundant=True)
    image_ids = data.read_image_ids(image_id_path)

    assert_sample_id_validity(sample_ids, image_ids)

    if model_args.features == 'grid':
        image_ds = data.GridFeaturesDataset(grid_features_path, image_ids)
    elif model_args.features == 'obj':
        image_md = data.read_image_metadata(obj_features_meta_path)
        image_ds = data.ObjectFeaturesDataset(obj_features_path, image_ids, image_md)
    else:
        raise ValueError(f'Invalid --features option: {model_args.features}')

    prediction_ids_cap = []
    prediction_ids_label = []
    prediction_results = []
    prediction_labels = []

    for sample_id in tqdm(sample_ids):
        # print(sample_id)
        features, locations = image_ds.read_data(sample_id)
        length = features.shape[0]

        if use_cuda:
            features = features.cuda()
            locations = locations.cuda()

        sample = {
            'net_input': {
                'src_tokens': features.unsqueeze(0),
                'src_locations': locations.unsqueeze(0),
                'src_lengths': [length]
            }
        }

        translations = task.inference_step(generator, models, sample)
        prediction = decode(captions_dict.string(translations[0][0]['tokens']))

        prediction_ids_cap.append(sample_id)
        prediction_ids_label.append(sample_id + 1)
        prediction_results.append(prediction)
        prediction_labels.append(label_preds)

    return pd.DataFrame.from_dict(data={
            'image_id': prediction_ids,
            'caption': prediction_results
        })

    if mode == 'captions':
        return pd.DataFrame.from_dict(data={
            'image_id': prediction_ids_cap,
            'caption': prediction_results
        })
    elif mode == 'labels':
        df = pd.DataFrame(data = prediction_labels, columns = classes)
        df.insert(0, "ImgId", prediction_ids_label, True)
        return df


def assert_sample_id_validity(sample_ids: iter, image_ids: iter):
    invalid_ids = [i for i in sample_ids if i not in image_ids]
    if len(invalid_ids) > 0:
        raise ValueError('Input sample ids {} are not present in the specified split.'.format(invalid_ids))


def print_predictions(predictions: pd.DataFrame, mode) -> None:
    print('Predictions:')
    print('============')
    if mode == 'captions':
        for sample_id, pred in predictions.to_numpy():
            print('{}: {}'.format(sample_id, pred))
    elif mode == 'labels':
    	print(predictions)

def store_predictions_as_csv(predictions: pd.DataFrame, file_path: str, mode) -> None:
    if mode == 'caption':
        print('\nWriting predictions to file "{}".'.format(file_path))
        predictions.to_json(file_path, orient='records')
    elif mode == 'labels':
        print('\nWriting predictions to file "{}".'.format(file_path))
        predictions.to_csv(file_path)

def cli_main():
    script_parser = get_script_parser()
    script_args, extra = script_parser.parse_known_args()

    parser = options.get_generation_parser(interactive=True, default_task='captioning')
    model_args = options.parse_args_and_arch(parser, input_args=extra)
    main(script_args, model_args)


def get_script_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'valid', 'test'], required=True,
                        help='The dataset split containing the samples provided in the input file (train|valid|test).')
    parser.add_argument('--output', type=str,
                        help='An optional output file used to store the captions predictions in json-format.')
    parser.add_argument('--output_l', type=str,
                        help='An optional output file used to store the labels predictions in csv-format.')
    parser.add_argument('--no-console-output', action='store_true',
                        help='Suppress printing the prediction results to the console.')
    return parser


if __name__ == '__main__':
    cli_main()
