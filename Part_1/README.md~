# Assignment 1

For the first assignment, the files in the tensor pack library were used.

These [files](https://github.com/ppwwyyxx/tensorpack) can be found in https://github.com/ppwwyyxx/tensorpack.

## Results
* The Zip created can be found in: https://drive.google.com/drive/folders/1-sTYxPe-EUABS27KFsi9FxKyQvqm75-n?usp=sharing
* Folders with at least 3000 imgaes of 4 frames were created in in the directory.
* Some examples are:
+ Alien-v0
	![Alien](./Alien-v0-0-19.png.png?raw=true)
+ Asteroids-v0
	![Ateroids](./Asteroids-v0-0-10.png?raw=true)
+ Breakout-v0
	![Breakout](./Breakout-v0-0-18.png?raw=true)
+ Pong-v0
	![Pong](./Pong-v0-0-16.png?raw=true)
+ Qbert-v0
	![Qbert](./Qbert-v0-0-34.png?raw=true)
+ Seaquest-v0
	![Seaquest](./Seaquest-v0-0-26.png?raw=true)
* To see more examples, go to the designated directory of each game in the zip file.



## What was done?
1. The first task was to be able to run the preloades examples provided.
In order to do so, it must be clarified that the section that this assignment work with was the A3C-Gym.
2. Using the terminal, in the directory of A3C-Gym, execute the command:
```
Python3 train-atari.py --task play --env Breakout-v0 --load Breakout-v0.npz
```
Where "Breakout-v0" is the name of the game, and "Breakout-v0.npz" is the name of the model.

3. Different games where tested. The [models](http://models.tensorpack.com/OpenAIGym/) of them were downloaded from the same page as provided.
The games tested were:
	* Alien-v0
	* Asteroids-v0
	* Breakout-v0
	* Pong-v0
	* Qbert-v0
	* Seaquest-v0

4. For each game, it was asked to save at least 3,000 observation instances, which consist on a concatenation of of 4 frames and outputs an action.
To do so, two .py files were modified.
- [ ] common.py.
It was modified bassically in its function of play_one_episode and play_n_episodes.
play_n_episodes was modified to receive the name of the game, and to have a counter of the iterations that play_one_episode will use.
	play_one_episode, in the other hand, was modified to save the frames previously mentioned into images, to an specific folder.
```
def play_n_episodes(game, player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    im_t = 0
    for k in range(nr):
        score = play_one_episode(k, im_t, game, player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))
```



```
def play_one_episode(k, im_t, game_name, env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        # Create the stacker
        stacker = np.empty((84, 0, 3), dtype="uint8")
        # Determine the action
        act = func(s[None, :, :, :])[0][0].argmax()
         # Create the image of the instace
        s, r, isOver, info = env.step(act)
        # First, verify if the game has not ended, to avoid looping
        if isOver:
            return act
        # For 4 frames
        for it in range(4):
            im = Image.fromarray(s[:, :, it*3:3*(it+1)])
            q = np.asarray(im)
            stacker = np.hstack((stacker, q))
        # Create the image
        im = Image.fromarray(stacker)
        im.save(game_name + "-" + str(k) + "-"+ str(im_t) + ".png")

        # Move the image in the designated folder
        current_path = game_name + "-" + str(k) + "-"+ str(im_t) + ".png"
        final_path = game_name + "_instances/" + current_path
        os.rename(current_path, final_path)

	# Random research. Uncomment to enable     
	# if random.random() < 0.01:
        #    spc = env.action_space
        #    act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        im_t = im_t+1
        if isOver:
            return sum_r
```
- [ ] train-atari.py.
	The modification it got was to generate a new folder, and obtain the game to pass it to the previous function in common.py.
```
        if args.task == 'play':
            # Obtain the name of the game
            game = str(args.env)
            # Create a folder inside the A3c-Gym folder for that game
            file_path = "/home/mlvm2/ce888labs/Assignment 1/tensorpack/examples/A3C-Gym/" + game + "_instances/"
            directory = os.path.dirname(file_path)
            # Verify if the folder exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Run the Game
            play_n_episodes(game, get_player(train=False), pred, args.episode, render=True)
```
