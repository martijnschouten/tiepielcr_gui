"""
.. module:: impedance
   :synopsis: This class describes the impedance format returned by the tiepieLCR
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

class impedance:
    value1 = 0
    value2 = 0
    error1 = 0
    error2 = 0
    timestamp = 0
    sig_offset = 0
    ref_offset = 0
    ref_clipped = False
    sig_clipped = False
    valid = False
    def set_values(self,new_value1,new_value2):
        """Set the impedance values of the object. What these values represents is determined by :func:`tiepieLCR_settings.tiepieLCR_settings.get_impedance_format`

        :param new_value1: The first impedance value. 
        :type new_value1: Complex double
        :param new_value2: The second impedance value.
        :type new_value2: Complex double
        :return: Nothing
        :rtype: None
        """
        self.value1 = new_value1
        self.value2 = new_value2

    def set_errors(self,new_error1,new_error2):
        """Set the impedance values of the object. What these values represents is determined by :func:`tiepieLCR_settings.tiepieLCR_settings.get_impedance_format`

        :param new_error1: The standard error in the first impedance value
        :type new_error1: Complex double
        :param new_error2: The standard error in the second impedance value
        :type new_error2: Complex double
        :return: Nothing
        :rtype: None
        """
        self.error1 = new_error1
        self.error2 = new_error2

    def set_timestamp(self,stamp,rel_stamp):
        """set the timestamp of the point in time the impedance was measured. This always is the timestamp of the last sample of the integration time.

        :param stamp: The timestamp
        :type stamp: Float
        :return: Nothing
        :rtype: None
        """
        self.rel_timestamp = rel_stamp
        self.timestamp = stamp

    def set_offsets(self,new_ref_offset,new_sig_offset):
        """Set the offset values of the object.

        :param new_ref_offset: The offset in the reference
        :type new_ref_offset: Double
        :param new_sig_offset: The offset in the signal
        :type new_sig_offset: Double
        :return: Nothing
        :rtype: None
        """
        self.sig_offset = new_sig_offset
        self.ref_offset = new_ref_offset

    def set_clipping(self,ref_clipping,sig_clipping):
        """Set wether or not the reference and signal clipped (=reached the maximum voltage that can be measured) during the integration time.

        :param ref_clipping: The reference clipped
        :type ref_clipping: Boolean
        :param sig_clipping: The signal clipped
        :type sig_clipping: Booelan
        :return: Nothing
        :rtype: None
        """
        self.ref_clipped = ref_clipping
        self.sig_clipped = sig_clipping
        if ref_clipping or sig_clipping:
            self.valid = False
        else:
            self.valid = True