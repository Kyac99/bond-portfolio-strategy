"""
Module pour la stratégie obligataire "Laddering".
"""

import pandas as pd
import numpy as np
from models.base_strategy import BondStrategy


class LadderingStrategy(BondStrategy):
    """
    Implémentation de la stratégie obligataire "Laddering".
    
    La stratégie Laddering consiste à répartir les investissements de manière équitable
    entre des obligations de différentes échéances, créant ainsi une "échelle" d'échéances.
    Cette approche permet de diversifier l'exposition aux taux d'intérêt et de réinvestir
    régulièrement à mesure que les obligations arrivent à échéance.
    """
    
    def __init__(self, maturities=None, weights=None, **kwargs):
        """
        Initialise une stratégie Laddering.
        
        Args:
            maturities (list, optional): Liste des maturités à utiliser (en années). 
                                         Par défaut [1, 2, 3, 5, 7, 10].
            weights (dict, optional): Poids à allouer à chaque maturité. 
                                     Par défaut, allocation égale.
            **kwargs: Arguments supplémentaires à passer à la classe parent.
        """
        super().__init__(name="Laddering Strategy", **kwargs)
        
        self.maturities = maturities if maturities else [1, 2, 3, 5, 7, 10]
        
        # Si aucun poids n'est spécifié, allouer de manière égale
        if weights is None:
            self.maturity_weights = {maturity: 1.0 / len(self.maturities) for maturity in self.maturities}
        else:
            # Normaliser les poids pour qu'ils totalisent 1
            total = sum(weights.values())
            self.maturity_weights = {maturity: weight / total for maturity, weight in weights.items()}
            
        # Dictionnaire pour associer chaque maturité à son ETF correspondant
        self.maturity_etf_map = {
            1: "SHY",    # iShares 1-3 Year Treasury Bond ETF (pour les maturités courtes)
            2: "SHY",    # Utiliser aussi SHY pour la maturité de 2 ans
            3: "IEI",    # iShares 3-7 Year Treasury Bond ETF
            5: "IEI",    # Utiliser aussi IEI pour la maturité de 5 ans
            7: "IEF",    # iShares 7-10 Year Treasury Bond ETF
            10: "IEF",   # Utiliser aussi IEF pour la maturité de 10 ans
            20: "TLT",   # iShares 20+ Year Treasury Bond ETF
            30: "TLT"    # Utiliser aussi TLT pour la maturité de 30 ans
        }
        
    def get_etf_for_maturity(self, maturity):
        """
        Retourne l'ETF à utiliser pour une maturité donnée.
        
        Args:
            maturity (int): Maturité en années.
            
        Returns:
            str: Symbol de l'ETF.
        """
        # Trouver la maturité la plus proche dans notre mapping
        closest_maturity = min(self.maturity_etf_map.keys(), key=lambda x: abs(x - maturity))
        return self.maturity_etf_map[closest_maturity]
    
    def generate_weights(self, data, date):
        """
        Génère les poids pour chaque ETF obligataire à une date donnée,
        en fonction de la stratégie de laddering.
        
        Args:
            data (pandas.DataFrame): Données des ETF obligataires.
            date (datetime): Date pour laquelle générer les poids.
            
        Returns:
            pandas.Series: Poids à allouer à chaque ETF.
        """
        # Si la date n'est pas dans les données, prendre la date la plus proche
        if date not in data.index:
            closest_date = data.index[data.index.get_indexer([date], method='nearest')[0]]
            date = closest_date
            
        # Initialiser les poids des ETF à zéro
        etf_weights = {etf: 0.0 for etf in set(self.maturity_etf_map.values())}
        
        # Distribuer les poids des maturités aux ETF correspondants
        for maturity, weight in self.maturity_weights.items():
            etf = self.get_etf_for_maturity(maturity)
            etf_weights[etf] += weight
            
        # Ne conserver que les ETF présents dans les données
        etf_weights = {etf: weight for etf, weight in etf_weights.items() if etf in data.columns}
        
        # Normaliser les poids pour qu'ils totalisent 1
        total = sum(etf_weights.values())
        if total > 0:  # Éviter la division par zéro
            etf_weights = {etf: weight / total for etf, weight in etf_weights.items()}
            
        return pd.Series(etf_weights)
    
    def adjust_for_yield_curve(self, yield_curve):
        """
        Ajuste les poids en fonction de la courbe des taux actuelle.
        
        En environnement de courbe pentue, favorise les maturités plus longues.
        En environnement de courbe plate ou inversée, favorise les maturités plus courtes.
        
        Args:
            yield_curve (pandas.Series): Courbe des taux avec les rendements par maturité.
            
        Returns:
            dict: Nouveaux poids ajustés par maturité.
        """
        # Calculer la pente de la courbe des taux (différence entre rendement à long terme et court terme)
        if '30Y' in yield_curve and '3M' in yield_curve:
            slope = yield_curve['30Y'] - yield_curve['3M']
        elif '10Y' in yield_curve and '3M' in yield_curve:
            slope = yield_curve['10Y'] - yield_curve['3M']
        else:
            # Si nous n'avons pas les points nécessaires, utiliser les poids par défaut
            return self.maturity_weights
        
        # Ajuster les poids en fonction de la pente
        adjusted_weights = {}
        
        if slope > 2.0:  # Courbe très pentue
            # Favoriser les maturités plus longues
            for maturity in self.maturities:
                if maturity <= 3:
                    adjusted_weights[maturity] = self.maturity_weights[maturity] * 0.7
                else:
                    adjusted_weights[maturity] = self.maturity_weights[maturity] * 1.3
        elif slope < 0:  # Courbe inversée
            # Favoriser les maturités plus courtes
            for maturity in self.maturities:
                if maturity <= 3:
                    adjusted_weights[maturity] = self.maturity_weights[maturity] * 1.5
                else:
                    adjusted_weights[maturity] = self.maturity_weights[maturity] * 0.5
        else:  # Courbe normale
            adjusted_weights = self.maturity_weights.copy()
            
        # Normaliser les poids pour qu'ils totalisent 1
        total = sum(adjusted_weights.values())
        adjusted_weights = {maturity: weight / total for maturity, weight in adjusted_weights.items()}
        
        return adjusted_weights
