// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Linq;

namespace CSharpUsage
{
    class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Using API");
            UseApi();
            Console.WriteLine("Done");
        }


        static void UseApi()
        {
            string modelName = "coreml_FNS-Candy_ImageNet.onnx";
            //string modelName = "rllib-dqn-pong_2019-04-02_17-28.onnx";
            //string modelName = "rllib-dqn-cartpole_2019-04-02_17-17.onnx";
            //string rlLibModelPath = Directory.GetCurrentDirectory() + @"\..\..\..\rllib-models\" + modelName;
            string rlLibModelPath = Directory.GetCurrentDirectory() + @"\..\..\..\odin-models\" + modelName;
            using (var cartPoleSession = new InferenceSession(rlLibModelPath))
            {
                var inputMeta = cartPoleSession.InputMetadata;
                var outputMeta = cartPoleSession.OutputMetadata;

                string inputName = inputMeta.Keys.First();
                int inputImageSize = 720 * 720 * 3;
                float[] inputImage = new float[inputImageSize];
                for(int x = 0; x < inputImageSize; x++)
                {
                    inputImage[x] = 1.0F;
                }
                if (inputMeta[inputName].Dimensions[0] == -1)
                {
                    inputMeta[inputName].Dimensions[0] = 1;
                }
                var tensor = new DenseTensor<float>(inputImage, inputMeta[inputName].Dimensions);
                var container = new List<NamedOnnxValue>();
                container.Add(NamedOnnxValue.CreateFromTensor<float>(inputName, tensor));
                using (var results = cartPoleSession.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    // dump the results
                    foreach (var r in results)
                    {
                        File.WriteAllText("image-results.txt", r.AsTensor<float>().GetArrayString());
                        //Console.WriteLine("Output for {0}", r.Name);
                        //Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
            }

            string modelPath = Directory.GetCurrentDirectory() + @"\squeezenet.onnx";

            // Optional : Create session options and set the graph optimization level for the session
            SessionOptions options = new SessionOptions();
            options.SetSessionGraphOptimizationLevel(2);

            using (var session = new InferenceSession(modelPath, options))
            {
                var inputMeta = session.InputMetadata;
                var outputMeta = session.OutputMetadata;
                var container = new List<NamedOnnxValue>();

                float[] inputData = LoadTensorFromFile(@"bench.in"); // this is the data for only one input tensor for this model

                foreach (var name in inputMeta.Keys)
                {
                    var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }

                // Run the inference
                using (var results = session.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                {
                    // dump the results
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output for {0}", r.Name);
                        Console.WriteLine(r.AsTensor<float>().GetArrayString());
                    }
                }
            }
        }

        static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            // read data from file
            using (var inputFile = new System.IO.StreamReader(filename))
            {
                inputFile.ReadLine(); //skip the input name
                string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                for (int i = 0; i < dataStr.Length; i++)
                {
                    tensorData.Add(Single.Parse(dataStr[i]));
                }
            }

            return tensorData.ToArray();
        }


    }
}
