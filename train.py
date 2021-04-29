import cv2
import neat
import numpy as np
import pickle
import retro


#função retirada de Lucas Thompson: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
def Dimensiona_input():

    inx_original, iny_original, _ = env.observation_space.shape
    inx = int(inx_original/8)
    iny = int(iny_original/8)

    return inx, iny


#função retirada de Lucas Thompson: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
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


def fitness(genomes, config):

    input_rede = [] #declaração da lista que será usada como input

    for genome_id, genome in genomes:

        ob = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        inx, iny = Dimensiona_input() #é feito uma redução da resolução da tela para que todas as partes da tela sejam enviadas como input da rede neural

        #atribuição de variáveis usadas no cálculo da fitness
        contador = 0
        fitness = 0
        
        moedas = 0
        moedas_max = 0
        x = 0
        x_max = 0 
        y = 0
        
        done = False

        while not done:

            env.render()            
            Gera_input(ob, input_rede, inx, iny) #a lista input_rede[] é preenchida com os valores da tela que serão usados de input        
            output_rede = net.activate(input_rede) #ativação da rede
            
            ob, rew, done, _ = env.step(output_rede) 
            ram = env.get_ram()
            Fecha_caixa(ram) #toda vez que o loop rodar, ele vai checar se a caixa de diálogo foi aberta, se sim, ele a fecha

            x = ram[0x95]*256 + ram[0x94] 
            #y = ram[0x97]*256 + ram[0x96]
             
            y = ram[0x00D3] #endereço obtido no do site SMW central
            moedas = ram[0x0DBF] #endereço obtido no do site SMW central
            powerup = ram[0x0019] #endereço obtido no site SMW central
            
            morto = ram[0x0071] #endereço obtido no discord da disciplina
            final_fase = ram[0x13D6] #endereço obtido no discord da disciplina
           
            pontos = rew/10 #ajusto o valor de reward para que a pontuação seja equilibrada em relação às demais formas de pontuar
            fitness += pontos
           
            if moedas > moedas_max: #a cada moeda que o mario coletar, ele ganha 50 pontos
                fitness += 50
                moedas_max = moedas
            
            if powerup != 0: #se o mario pegar o powerup e ficar grande, ele ganha 1 ponto a cada frame
                fitness += 1
        
            if y < 96: #checo a posição y do mário, se ela for menor que 96, significa que ele pulou ou está num terreno elevado
                fitness += 1 
               
            if x > x_max: #checo se o Mario está andando pra direita
                fitness += 2
                x_max = x
                contador = 0 #zero o contador caso o Mario esteja andando pra direita
            
            if x == x_max or x < x_max: #checo se o Mario está parado ou andando pra esquerda
                contador += 1 #conto quantos frames o Mario ficou nessa condição
                fitness -= 1 #penalizo o fitness da rede neural
                    
            if morto == 9: #penalizo o Mario por morrer
                fitness -= 200
                done = True   #o loop para de rodar
                
                print('Terminou de jogar!')
                print("ID da rede:", genome_id)
                print ("Fitness obtida:", fitness)
                print('Motivo da finalização: morte')
                print('')
                
            if final_fase < 80 and morto == 0:
                fitness = 25000 #a rede neural que chegar no final da fase ganha o máximo de pontos de fitness
                done = True
                
                print('Terminou de jogar!')
                print("ID da rede:", genome_id)
                print ("Fitness obtida:", fitness)
                print('Motivo da finalização: finalizou a fase')
                print('')
                
            if contador == 250: #250 frames é aproximadamente 5 segundos do relógio do jogo
                done = True #se o contador chegar ao valor de 250, a variável done recebe o valor True e é encerrada aquela jogada
                
                print('Terminou de jogar!')
                print("ID da rede:", genome_id)
                print ("Fitness obtida:", fitness)
                print('Motivo da finalização: parou de avançar na fase')
                print('')
                    
            genome.fitness = fitness
            

def main():
    
    print('')
    print("Para começar um novo treino digite 0, se não, digite o número de 1 a 25 referente ao checkpoint que deseja continuar o treino")
    digito = input()

    if digito == '0':
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'configuracoes.txt')
        pop = neat.population.Population(config)
    else: 
        pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-' + digito)
    
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(1))

    rede_treinada = pop.run(fitness)

    with open('rede_treinada.pkl', 'wb') as treinada:
        pickle.dump(rede_treinada, treinada)

        
if __name__ == "__main__":
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1)
    main()