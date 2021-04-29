import cv2
import neat
import numpy as np
import pickle
import retro


#função retirada de Lucas Thompson
def Dimensiona_input():

    inx_original, iny_original, _ = env.observation_space.shape
    inx = int(inx_original/8)
    iny = int(iny_original/8)

    return inx, iny


#função retirada de Lucas Thompson
def Gera_input(ob, imgarray, inx, iny):

    imgarray.clear()

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))

    for x in ob:
        for y in x:
            imgarray.append(y)

    return imgarray


#função feita para fechar a caixa de diálogo aberta pelo Mario
def Fecha_caixa(ram):
    
    if ram[0x1426] != 0:
        aperta_A = [1, 1, 1, 1, 1, 1, 1, 1, 1] #neste caso ele aperta todos os botões pois nem sempre funcionava mandar apenas o comando do botão A
        env.step(aperta_A)
        #print("apertou A")
    
    
def play(genomes, config):

    input_rede = [] #declaração da lista que será usada como input

    for genome in genomes:

        ob = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        inx, iny = Dimensiona_input() #é feito uma redução da resolução da tela para que todas as partes da tela sejam enviadas como input da rede neural

        done = False

        while not done:

            env.render()            
            Gera_input(ob, input_rede, inx, iny) #a lista input_rede[] é preenchida com os valores da tela que serão usados de input        
            output_rede = net.activate(input_rede) #ativação da rede
            
            ob, rew, done, info = env.step(output_rede) 
            ram = env.get_ram()
            Fecha_caixa(ram) #toda vez que o loop rodar, ele vai checar se a caixa de diálogo foi aberta, se sim, ele a fecha
            
            final_fase = ram[0x13D6] #endereço obtido no discord da disciplina
           
            if final_fase < 80:
                fitness = 25000 #a rede neural que chegar no final da fase ganha o máximo de pontos de fitness
                done = True
                
                print('')
                print('Terminou de jogar!')
                print ("Fitness obtida:", fitness)
                print('Motivo da finalização: finalizou a fase')


def main():
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configuracoes.txt')

    with open('rede_treinada.pkl', 'rb') as arquivo:
        treinada = pickle.load(arquivo)

    redes = [treinada]
    play(redes, config)

if __name__ == "__main__":
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    main()