"""beom_dataset dataset."""

import os
import uuid
import json
import tensorflow as tf
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for beom_dataset dataset."""

  VERSION = tfds.core.Version('1.0.8')
  RELEASE_NOTES = {
      '1.0.8': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description="_DESCRIPTION",
        features=tfds.features.FeaturesDict({
            "text": tfds.features.Text(),
        }),
        citation="_CITATION",
        homepage="https://github.com/google-research/text-to-text-transfer-transformer#datasets",
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    path = dl_manager.manual_dir
    return {
        'train': self._generate_examples(path),
    }
 
  def _generate_examples(self, file_dir: str):
    beam = tfds.core.lazy_imports.apache_beam

    def _process_lines(line):
      return str(uuid.uuid4()), {'text': json.loads(line.strip())['text']}
      
    return (
        beam.Create(tf.io.gfile.listdir(file_dir))
         | beam.combiners.Sample.FixedSizeGlobally(3)
         | beam.FlatMap(lambda x: x)
         | beam.Map(lambda x: os.path.join(file_dir, x))
         | beam.io.textio.ReadAllFromText()
         | beam.combiners.Sample.FixedSizeGlobally(5)
         | beam.FlatMap(lambda x: x)
         | beam.Map(_process_lines)
    )

  
'''
tfds build \
--manual_dir=gs://lg-maxtext/test-dataset1 \
--data_dir=gs://dataset-temp-dataflow/beom/test1 \
--beam_pipeline_options=\
"runner=DirectRunner,"\
"project=lg-air-pso,"\
"region=us"\
"job_name=beom-test-job,"\
"staging_location=gs://dataset-temp-dataflow/beom/binaries,"\
"temp_location=gs://dataset-temp-dataflow/beom/temp,"\
"requirements_file=/home/jupyter/beom_dataset/beam_requirements.txt"
'''