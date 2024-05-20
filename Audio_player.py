import pygame
from pygame import mixer

def play_audio(filename):
    # Initialize Pygame mixer module
    mixer.init()

    # Load the audio file
    audio = mixer.Sound(filename)

    # Play the audio file
    audio.play()
    print("............................................played audio..")

    # Wait for the desired duration before stopping the audio
    duration = 14 # in seconds
    pygame.time.wait(duration * 1000)

    # Stop the audio
    audio.stop()
        
    
            
# play_audio('SBH_R_ENGLISH.mp3')

# import pygame

# # Initialize Pygame mixer module
# pygame.mixer.init()

# # Load the audio file
# audio = pygame.mixer.Sound('SBH_R_BENGALI.mp3')

# # Play the audio file
# audio.play()

# # Keep the program running until the audio finishes playing
# while pygame.mixer.get_busy():
#     pygame.time.Clock().tick(10)