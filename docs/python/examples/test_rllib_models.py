import os
import math
import onnxruntime as rt
import numpy
import gym
import _pickle as pickle
import tensorflow as tf
from tensorflow.python.client import session
from onnxruntime.datasets import get_example

base_path = os.path.abspath(os.path.dirname(__file__))

def get_onnx_session(model_path):
    # example1 = get_example("sigmoid.onnx")
    #sess = rt.InferenceSession(example1)
    # sess = rt.InferenceSession("C:\\Src\\onnxruntime\\csharp\\sample\\Microsoft.ML.OnnxRuntime.InferenceSample\\rllib-models\\rllib-ppo-cartpole.onnx")
    sess = rt.InferenceSession(model_path)
    #sess = rt.InferenceSession("rllib-ppo-cartpole.onnx")

    for input_metadata in sess.get_inputs():
        print("input name", input_metadata.name)
        print("input shape", input_metadata.shape)
        print("input type", input_metadata.type)
    input_name = sess.get_inputs()[0].name
    local_input = sess.get_inputs()[0]

    for output_metadata in sess.get_outputs():
        print("output name", output_metadata.name)
        print("output shape", output_metadata.shape)
        print("output type", output_metadata.type)    
    
    return sess

def get_tf_session(model_path):
    # import tensorflow as tf
    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    g = tf.Graph()
    # with g.as_default():
    #     with tf.Session(graph=g) as sess:
    sess = tf.Session(graph=g)
    meta_graph_def = \
        tf.saved_model.loader.load(sess,
                            [tf.saved_model.tag_constants.SERVING],
                            model_path)
    print("Signature Def Information:")
    print(meta_graph_def.signature_def[signature_key])
    return sess    

frozen_session_input = None
frozen_session_output = None
def get_tf_frozen_session(model_path, algo):
    # load the protobuf file, parse it to retrieve the unserialized graph_def
    frozen_graph_filename = os.path.join(model_path,"saved_model.pb")
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # import the graph_def into a new Graph and return it 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    for op in graph.get_operations():
            print(op.name)
            # prefix/Placeholder/inputs_placeholder
            # ...
            # prefix/Accuracy/predictions    return graph

    # We access the input and output nodes 
    global frozen_session_input, frozen_session_output
    if algo == "DQN":
        frozen_session_input = graph.get_tensor_by_name('import/default/obs_input:0')
        frozen_session_output = graph.get_tensor_by_name('import/default/cond/Merge:0')
    elif algo == "PPO":
        frozen_session_input = graph.get_tensor_by_name('import/default/obs_input:0')
        frozen_session_output = graph.get_tensor_by_name('import/default/Squeeze:0')

    sess = tf.Session(graph=graph)
    return sess

def onnx_session_evaluator(session, local_obs, algo):
    if algo=="DQN":
        feed_dict = {"default/obs_input:0": local_obs,
                #    "default/eps:0": float(0.020000000000000018),
                   "default/eps:0": float(0.0),
                   "default/q_func/PlaceholderWithDefault:0": False,
                   "default/stochastic:0": False
        }
        output_key = "default/cond/Merge:0"
    elif algo=="PPO":
        feed_dict = {"default/obs_input:0": local_obs}
        output_key = "default/Squeeze:0"

    inf_result = session.run([output_key], feed_dict)
    return inf_result[0][0]

def tf_session_evaluator(session, local_obs, algo):
    if algo=="DQN":
        feed_dict = {"default/obs_input:0": local_obs,
                #    "default/eps:0": float(0.020000000000000018),
                   "default/eps:0": float(0.0),
                   "default/q_func/PlaceholderWithDefault:0": False,
                   "default/stochastic:0": False
        }
        output_key = "default/cond/Merge:0"
    elif algo=="PPO":
        feed_dict = {"default/obs_input:0": local_obs}
        output_key = "default/Squeeze:0"

    inf_result = session.run(output_key, feed_dict=feed_dict)
    return inf_result[0]

def tf_frozen_session_evaluator(session, local_obs, algo):
    if algo == "DQN":
        feed_dict = {frozen_session_input: local_obs,
                #    "default/eps:0": float(0.020000000000000018),
                   "import/default/eps:0": float(0.0),
                #    "import/default/q_func/PlaceholderWithDefault:0": False,
                   "import/default/stochastic:0": False
        }
        inf_result = session.run(frozen_session_output, feed_dict=feed_dict)
    elif algo == "PPO":
        inf_result = session.run(frozen_session_output, feed_dict={frozen_session_input: local_obs})

    return inf_result[0]


def evaluate_model(onnx_session, tf_session, episode_count, onnx_session_evaluator, tf_session_evaluator, algo, use_onnx = True, render_env = False):
    # x = numpy.random.random((3,4,5))
    # x = x.astype(numpy.float32)

    #env = gym.make("SpaceInvaders-v0")
    print("Evaluating model from framework:{} trained with algo:{}".format("onnx" if use_onnx else "tf", algo))
    env = gym.make("CartPole-v0")
    for i_episode in range(episode_count):
        observation = env.reset()
        done = False
        reward_total = 0.0
        for i_step in range(200):
            if render_env:
                env.render()
            # print(observation)
            local_obs = numpy.array([observation]).astype(numpy.float32)
            # local_obs = numpy.random.random((None, 4))
            # inf_result = onnx_session.run(["default/Squeeze:0"], {"default/obs_input:0": local_obs})
            # print("inf_result: " + str(inf_result[0][0]))
            # action = inf_result[0][0]
            if use_onnx:
                onnx_action = onnx_session_evaluator(onnx_session, local_obs, algo)
            else: 
                tf_action = tf_session_evaluator(tf_session, local_obs, algo)
            # print("action:{}".format(action))
            # print("onnx_action", onnx_action, "tf_action", tf_action)
            # random_action = env.action_space.sample()

            local_action = onnx_action if use_onnx else tf_action
            local_output = [local_obs[0][0], local_obs[0][1], local_obs[0][2], local_obs[0][3], local_action]
            # print (",".join(str(value) for value in local_output))
            observation, reward, done, info = env.step(local_action)
            reward_total += reward
            position = observation[0]
            angle_theta = observation[2]
            # angle_threshold = 36
            # angle_theta_threshold = angle_threshold * 2 * math.pi / 360
            # local_done = position > 2.4 or position < -2.4 or angle_theta > angle_theta_threshold or angle_theta < -angle_theta_threshold
            # local_done = bool(local_done)
            # if local_done:
            if done:
                print("Episode {} finished after {} timesteps with reward {}.  Position:{} Angle:{}".format(i_episode + 1, i_step + 1, reward_total, position, angle_theta))
                break
    env.close()

def compare_model(model_path, tf_session, tf_session_evaluator, baseline_path):
    print("Comparing TF model from {} to baseline session at {}".format(model_path, baseline_path))
    with open(baseline_path, 'rb') as baseline_pkl:
        baseline_blob = pickle.load(baseline_pkl)
        baseline_data = baseline_blob[0]
        step_index = 0
        for baseline_step in baseline_data:
            state = baseline_step[0]
            print("baseline_step[{}] state:{} action:{} next_state:{} reward:{} done:{}".format(step_index, 
                state, baseline_step[1], baseline_step[2], baseline_step[3], baseline_step[4]))
            model_input = numpy.array([state]).astype(numpy.float32)
            model_action = tf_session_evaluator(tf_session, model_input)
            print("model_step[{}] state:{} action:{}".format(step_index, state, model_action))
            step_index += 1
    print("Finished model comparison to baseline")



if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach()
    # print("Waiting for debugger to attach")
    # ptvsd.wait_for_attach() # (Optional) This blocks until a debugger has attached

    # onnx_model_path = "C:\\Src\\onnxruntime\\csharp\\sample\\Microsoft.ML.OnnxRuntime.InferenceSample\\rllib-models\\rllib-ppo-cartpole.onnx"
    # onnx_model_path = os.path.join(base_path, "onnx_models", "rllib-ppo-cartpole.onnx")
    test_path = os.path.join(base_path, "tf_models", "rllib-dqn-cartpole_2019-06-06_01", "onnx_model", "rllib-dqn-cartpole_2019-06-06_01.onnx")
    path_exists = os.path.exists(test_path)
    onnx_model_path = os.path.join(base_path, "tf_models", "rllib-dqn-cartpole_2019-06-06_01", "onnx_model", "rllib-dqn-cartpole_2019-06-06_01.onnx")
    onnx_session = get_onnx_session(onnx_model_path)
    # onnx_session = None

    # tf_model_path = "C:\\Src\\athens02\\tests\\unit_tests\\assets\\rllib\\rllib-ppo-cartpole_2019-04-10_16-32"
    # tf_model_path = os.path.join(base_path, "tf_models", "rllib-ppo-cartpole_2019-06-10_01", "export_model")
    tf_model_path = os.path.join(base_path, "tf_models", "rllib-ppo-cartpole_2019-06-18_01", "frozen_model")
    # tf_session = get_tf_session(tf_model_path)
    tf_session = get_tf_frozen_session(tf_model_path, "PPO")

    # baseline_path = os.path.join(base_path, "tf_models", "rllib-dqn-cartpole_2019-05-01_00-31", "rllib-rollout-200steps.pkl")
    # compare_model(tf_model_path, tf_session, tf_session_evaluator, baseline_path)
    # exit(0)

    # evaluate_model(onnx_session, 10, onnx_session_evaluator)
    # evaluate_model(onnx_session, tf_session, 10, onnx_session_evaluator, tf_session_evaluator, True)
    tf_evaluator = tf_frozen_session_evaluator
    # tf_evaluator = tf_session_evaluator
    evaluate_model(onnx_session, tf_session, 10, onnx_session_evaluator, tf_evaluator, "DQN", use_onnx=True, render_env=False)
    tf_session.close()
