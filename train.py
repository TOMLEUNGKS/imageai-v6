from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("train_image")
model_trainer.trainModel(num_objects=6, num_experiments=500, enhance_data=True, batch_size=20, show_network_summary=True)
