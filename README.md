# axelrod-ML
Rather than query dozens of game theorists for strategies like Axelrod did, what if we build simple neural net models to play strategies in an iterated prisoner's dilemna? What if they can evolve?

## What's here?
We've implemented a iterated prisoner's dilemna and a number of strategies to deploy. Strategies compete with each other and the losing strategies can be removed and the successful ones copied to evolve the strategies, and subsequently the environment, over time. Additionally, some strategies can be copied with mutations to allow for changes within the strategy space and not just in the overall environment.

## Getting started
We're still getting started too. Currently, running game.py will let you play your own strategy against the famous "tit-for-tat". The UI for tournament is still under development, though currently it runs a tournament with a set of actors whose strategy is dictated by a small recurrent neural net. Each iteration removes losers and copies a winner with slight mutations in the strategy.

## TODO
Plenty
